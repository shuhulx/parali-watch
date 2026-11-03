# AI assistance

I used **Claude Code and OpenAI Codex** for implementation, debugging, code review and documentation. Prompts covered the data-to-decision pipeline, accuracy improvements, reproducibility checks and walkthrough preparation.

My inputs included the working reference number, the submission scope and requests for review. During final review, I questioned whether one failed satellite scene should abort the entire run. This led to restoring partial scene processing with warnings and cached successes; evaluation still requires the imagery needed for its reference dates. I also asked to keep the review narrow and avoid unnecessary model changes. Thresholds, features and training labels were left unchanged during that review.

Claude vision models produced the training and reference annotations. These are AI interpretations, not field-verified labels; the model card records their uncertainty and the evaluation limits.
