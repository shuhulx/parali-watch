# AI assistance

I used **Claude Code** for design discussion, implementation, debugging and pipeline runs. I requested **OpenAI Codex** reviews to check the code and evaluation; Codex also implemented the field-voting experiment, which was later disabled, and helped with the final documentation and dashboard polish.

The prompts focused on completing the data-to-decision pipeline, improving burn detection, checking training/test separation and preparing explanations for the live walkthrough. My inputs included the working reference number 123456, the accuracy goals, requests for independent review and the final submission scope. Code changes and verification were carried out with the assistants.

Claude vision models interpreted the satellite chips used for training and evaluation labels. Those labels are AI photo-interpretation, not field-verified ground truth. The model card records the labelling method, rejected experiments and evaluation limits.
