import sys
import unittest

from contracts import HostPageOwner, Identity, ContractError, MLA_LAYERS
from reference import stage_reference


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.page_bytes = 256 * 544 * 11
        self.owner = HostPageOwner(namespace="model/profile/session", layout_epoch=1,
                                   budget_bytes=2*self.page_bytes,
                                   indexer_capacity_tokens=4096)
        self.identity = Identity(self.owner.namespace, b"hash-A", 0, 1, 1)

    def publish(self, identity=None):
        identity = identity or self.identity
        nonce = self.owner.begin_spill(identity)
        for layer in range(MLA_LAYERS):
            self.owner.queued(nonce, layer)
        for layer in range(MLA_LAYERS):
            self.owner.completed(nonce, layer)
        return nonce

    def test_resident_admission_does_not_allocate_host(self):
        self.assertEqual(self.owner.admission(GPU_prefix_fits=True,
            required_host_pages=500, requested_tokens=2048), "resident")
        self.assertEqual(self.owner.reserved_bytes, 0)

    def test_budget_and_gpu_indexer_capacity_fail_before_spill(self):
        for pages, tokens in [(3, 2048), (1, 4097)]:
            with self.assertRaises(ContractError):
                self.owner.admission(GPU_prefix_fits=False,
                                      required_host_pages=pages, requested_tokens=tokens)
        self.assertFalse(self.owner.pending)

    def test_enqueued_is_not_published(self):
        nonce = self.owner.begin_spill(self.identity)
        for layer in range(10): self.owner.queued(nonce, layer)
        self.assertFalse(self.owner.source_releasable_in_stream_order(nonce))
        self.owner.queued(nonce, 10)
        self.assertTrue(self.owner.source_releasable_in_stream_order(nonce))
        self.assertNotIn(self.identity, self.owner.valid)
        for layer in range(10): self.owner.completed(nonce, layer)
        self.assertNotIn(self.identity, self.owner.valid)
        self.assertEqual(self.owner.completed(nonce, 10), "published")

    def test_cancelled_partial_spill_retains_destination_until_fence(self):
        nonce = self.owner.begin_spill(self.identity)
        self.owner.queued(nonce, 2)
        self.assertEqual(self.owner.cancel_spill(nonce), "drain_required")
        self.assertEqual(self.owner.reserved_bytes, self.page_bytes)
        with self.assertRaises(ContractError): self.owner.queued(nonce, 3)
        self.assertEqual(self.owner.completed(nonce, 2), "discarded_after_completion")
        self.assertEqual(self.owner.reserved_bytes, 0)

    def test_cancel_before_enqueue_reclaims_without_cuda_assumption(self):
        nonce = self.owner.begin_spill(self.identity)
        self.assertEqual(self.owner.cancel_spill(nonce), "discarded_before_enqueue")
        self.assertEqual(self.owner.reserved_bytes, 0)

    def test_no_step_reads_pending_pages(self):
        self.owner.begin_spill(self.identity)
        with self.assertRaises(ContractError):
            self.owner.begin_step(identities=(self.identity,), start_position=256, query_rows=8)

    def test_stale_namespace_generation_and_layout_rejected(self):
        for i in [Identity("other", b"h", 0, 1, 1),
                  Identity(self.owner.namespace, b"h", 0, 0, 1),
                  Identity(self.owner.namespace, b"h", 0, 1, 2)]:
            with self.assertRaises(ContractError): self.owner.begin_spill(i)

    def test_cancelled_read_defers_reuse_and_does_not_publish(self):
        self.publish()
        step = self.owner.begin_step(identities=(self.identity,), start_position=256, query_rows=8)
        self.assertEqual(self.owner.retire(self.identity), "deferred_until_step_completion")
        self.owner.cancel_step(step.nonce)
        with self.assertRaises(ContractError):
            self.owner.finish_step(step.nonce, actual_event_complete=False,
                                   checked_kernel_error_mask=0, accepted_tokens=8)
        self.assertIn(self.identity, self.owner.valid)
        with self.assertRaises(ContractError):
            self.owner.finish_step(step.nonce, actual_event_complete=True,
                                   checked_kernel_error_mask=0, accepted_tokens=8)
        self.assertEqual(self.owner.visible_position, 0)
        self.assertNotIn(self.identity, self.owner.valid)

    def test_acceptance_hides_rejected_rows(self):
        self.publish()
        step = self.owner.begin_step(identities=(self.identity,), start_position=256, query_rows=8)
        self.assertEqual(self.owner.finish_step(step.nonce, actual_event_complete=True,
            checked_kernel_error_mask=0, accepted_tokens=3), 259)
        with self.assertRaises(ContractError):
            self.owner.finish_step(step.nonce, actual_event_complete=True,
                                   checked_kernel_error_mask=0, accepted_tokens=8)

    def test_kernel_errors_and_mismatched_resume_epochs_do_not_commit(self):
        self.publish()
        for flags, target, draft in [(4, 1, 1), (0, 2, 1)]:
            step = self.owner.begin_step(identities=(self.identity,), start_position=256, query_rows=8)
            with self.assertRaises(ContractError):
                self.owner.finish_step(step.nonce, actual_event_complete=True,
                    checked_kernel_error_mask=flags, accepted_tokens=8,
                    target_checkpoint_epoch=target, paired_draft_checkpoint_epoch=draft)
            self.assertEqual(self.owner.visible_position, 0)


class PackedReferenceTests(unittest.TestCase):
    def metadata(self):
        # Nonidentity source pages and deliberately distinguishable raw scales.
        return dict(source_pages=(1, 0),
            physical_packed_rows=tuple(bytes([n % 251]) * 512 for n in range(512)),
            physical_scale_rows=tuple(bytes([n % 241]) * 32 for n in range(512)),
            slot_generations=(7, 9), expected_generations=(9, 7))

    def test_q8_union_includes_tail_duplicates_masks_and_order(self):
        rows = tuple(tuple([4*r, 4*r+1, 4*r+2, 4*r+3,
                            257, 258, 259, 257] + [-1]*24) for r in range(8))
        result = stage_reference(rows, row_visible_limits=(512,)*8, **self.metadata())
        self.assertEqual(len(result.logical_to_hot), 35)
        for r in range(8):
            for col, logical in enumerate(rows[r]):
                slot = result.remapped_indices[r][col]
                if logical == -1:
                    self.assertEqual(slot, -1)
                else:
                    physical = (1 if logical < 256 else 0)*256 + logical%256
                    self.assertEqual(result.packed_rows[slot], bytes([physical%251])*512)
                    self.assertEqual(result.scale_rows[slot], bytes([physical%241])*32)
        self.assertEqual(result.remapped_indices[0][4], result.remapped_indices[7][7])

    def test_q1_no_selection_is_not_fake_data(self):
        result = stage_reference(((-1,)*32,), row_visible_limits=(512,), **self.metadata())
        self.assertFalse(result.logical_to_hot)
        self.assertEqual(result.remapped_indices, ((-1,)*32,))

    def test_stale_generation_future_row_and_missing_page_fail(self):
        kwargs = self.metadata()
        for changes, first in [({'expected_generations':(8,7)}, 1),
                               ({'source_pages':(-1,0)}, 1), ({},512)]:
            with self.assertRaises(ContractError):
                stage_reference(((first,)+(-1,)*31,), row_visible_limits=(512,),
                                **(kwargs | changes))

    def test_budget_shape_bounds_fail(self):
        for rows, slots in [((tuple(range(32)),)*9, 16640),
                            (((1,)*32,), 31), (((1,)*31,),16640)]:
            with self.assertRaises(ContractError):
                stage_reference(rows, max_slots=slots,
                    row_visible_limits=(512,)*len(rows), **self.metadata())

    def test_separate_layers_do_not_share_packed_payload(self):
        a = stage_reference(((1,)*32,), row_visible_limits=(512,), **self.metadata())
        meta = self.metadata()
        meta['physical_packed_rows'] = tuple(bytes([88])*512 for _ in range(512))
        b = stage_reference(((1,)*32,), row_visible_limits=(512,), **meta)
        self.assertNotEqual(a.packed_rows, b.packed_rows)


class DisabledImportTests(unittest.TestCase):
    def test_explicit_default_off_without_torch_import(self):
        from selected_host_q8 import SelectedHostQ8
        self.assertNotIn('torch', sys.modules)
        with self.assertRaises(RuntimeError):
            SelectedHostQ8(None,None,None,device_index=0,max_host_bytes=0,max_staging_bytes=0)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
