# AI assistance

I used Claude Code as the main development assistant, with additional AI assistance for code review, testing and submission polish. The assistance covered pipeline implementation, debugging, documentation and explanations of the technical choices.

My contribution centred on directing the work and setting priorities. I chose 123456 as the working reference and asked for a complete implementation of the resulting configuration, with architectural choices I could follow and explain. I pushed for better detection accuracy and requested separate reviews of the code and evaluation. During refinement, I kept the scope on the existing pipeline and asked for focused improvements rather than a rewrite.

Typical prompts asked the assistants to check coverage against the assignment, investigate weak detection results, justify modelling choices, and examine the training split for leakage. Before submission, I also requested a development-set error review to establish whether further accuracy changes were justified. These are summaries of the requests, not verbatim quotations.

Claude vision models interpreted satellite-image chips to generate the training and reference labels. Those labels have not been field verified; their uncertainty and the evaluation limits are documented in the model card.
