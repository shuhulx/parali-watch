# AI assistance

For the final review, I asked to keep the existing pipeline intact and limit changes to what was needed for submission. I challenged a change that would stop the entire run if one satellite image failed to download. This led to restoring partial processing, with failures reported and successful downloads kept for retry.

I used Claude Code for implementation and debugging, with additional AI coding and review assistance for the final checks and documentation.

Claude vision models generated the training and reference labels. Their uncertainty and lack of field verification are documented in the model card.
