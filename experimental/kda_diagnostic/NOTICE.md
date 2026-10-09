# License and external source boundary

These diagnostic tools, tests, fixtures, and documentation are supplied under this repository's MIT license. The base repository retains its original Flun copyright and license unchanged.

The synthetic request fixtures contain constructed arithmetic, counting, inventory, and concurrency-analysis prompts. They contain no user conversation, private corpus, model weights, or host configuration.

Tests read selected methods and declarations from an external, user-supplied ExLlamaV3 source tree. ExLlamaV3 is a separate project by turboderp and contributors; its own license and attribution apply to that external source and compiled extension. No substantial external engine source, native header/library, or model binary is redistributed here. Source pins identify the required revision but do not confer ABI or numerical compatibility.
