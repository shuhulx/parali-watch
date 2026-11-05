# AI assistance

I used Claude Code as the main development assistant for pipeline implementation, debugging and explanations of the technical choices, as well as code review, testing and documentation.

My contribution centred on directing the work and setting priorities. I chose 123456 as the working reference and asked for a complete implementation of the resulting configuration, with architectural choices I could follow and explain. I pushed for better detection accuracy and requested separate reviews of the code and evaluation. During refinement, I kept the scope on the existing pipeline and asked for focused improvements rather than a rewrite.

Typical prompts asked the assistants to check coverage against the assignment, investigate weak detection results, justify modelling choices, and examine the training split for leakage. Before submission, I also requested a development-set error review to establish whether further accuracy changes were justified. These are summaries of the requests, not verbatim quotations.

For labelling, Claude Sonnet and Opus used their image-analysis capabilities to interpret before-and-after satellite-image chips and produce training and reference labels. These labels have not been field verified. The model card documents their uncertainty and the limits of evaluating against them.
