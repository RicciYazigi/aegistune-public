# Security scope

AegisTune is a synthetic research simulator. Its tools do not send messages,
make HTTP webhook deliveries or connect to real payment rails. The in-process
sandbox is not a security boundary against arbitrary Python code execution.

Ticket/model text is untrusted. The offline viewer renders it as text and escapes
embedded JSON script content. Dataset identity, strict joins and artifact hashes
detect content mismatches; they are not cryptographic signatures or a universal
proof of model safety.

Do not submit credentials or real customer/financial information in public issues
or pull requests. If private vulnerability reporting is enabled on the published
repository, use it for sensitive details. Otherwise, disclose only a nonsensitive
summary while arranging an appropriate private channel. No response-time or
production-support commitment is offered by this research release.
