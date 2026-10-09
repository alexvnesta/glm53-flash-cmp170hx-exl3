"""Drain an owned executor/task despite caller cancellation. Standard library only."""
import asyncio


async def drain_owned_future(future):
    cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            cancelled = True
    result = future.result()
    if cancelled:
        raise asyncio.CancelledError
    return result
