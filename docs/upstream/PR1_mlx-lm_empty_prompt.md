# PR 1 → ml-explore/mlx-lm: server: reject empty prompts instead of ending the generation thread

Branch: `server-reject-empty-prompt` (patch: `0001-server-reject-empty-prompts-instead-of-ending-the-ge.patch`, on main 56230fa)

## Body
An empty tokenized prompt crashes the batched server.
- **Trigger:** `"prompt": ""` with a tokenizer that adds no BOS token.
- **Failure:** the request reaches `BatchGenerator.insert_segments`, which raises `ValueError("Sequence ... has an empty prompt.")` inside the generation thread.
- **Effect:** the thread exits and every later request fails, until the server is restarted.

This PR:
- rejects the request before it is inserted, the same way #1935 does for `max_tokens: 0`;
- returns 400 for `ValueError`s raised while setting up generation (404 before).

Related: #1909 / #1513 (the general "keep the loop alive" fix). This PR is the narrow fix for one trigger.

Test: `test_empty_prompt_is_rejected_and_server_survives`. It sends an empty prompt and expects 400, then sends a normal request and expects 200.
