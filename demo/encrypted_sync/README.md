# Encrypted sync Spike A

This runnable spike demonstrates a narrow architecture claim:

> Real encryption, real relay. Conflict merge and key rotation are the next
> two layers.

Two agent containers keep separate recall stores. They share no Docker
network and no data volume. A blind relay is the only service attached to
both internal networks. Agent A exports recall's shipped portable archive,
encrypts it with age X25519, and uploads only the ciphertext. Agent B
downloads, decrypts, and merge-imports that same archive.

Run the evidence sequence:

```sh
python demo/encrypted_sync/run_demo.py
```

The sequence proves each claim by fruit:

1. Agent B retrieves a local control fact, then fails to find the target fact.
2. Agent A saves the target, exports, encrypts, and uploads it.
3. Agent B downloads, decrypts, merge-imports, and retrieves the target.
4. The relay's volume is scanned for the plaintext. The same grep first
   detects a planted control on standard input, then reports zero volume hits.
5. The relay stops. Agent B still retrieves the imported fact locally.
6. A runtime DNS probe confirms agent A cannot resolve agent B. The stronger
   topology evidence is the compose file itself: the agents have disjoint
   internal networks and only the relay is dual-homed.

Use `--json` for machine-readable evidence and `--keep` to retain the
containers for inspection. The Docker integration test runs with:

```sh
SYNAPT_RUN_ENCRYPTED_SYNC_DOCKER=1 \
  python -m pytest tests/integrations/test_encrypted_sync_spike.py -q
```

## Deliberate limits

- Merge is last-write-wins by a monotonically increasing logical clock. This
  is not the CRDT layer.
- The two agents use one pre-shared team identity generated for each run.
  Membership-driven key rotation is not demonstrated.
- The relay sees object sizes, ordering, and timing. It does not receive the
  decryption identity or plaintext archive.
- Containers on one host share a kernel. This proves the data path and the
  relay's inability to read stored blobs. It does not prove resistance to a
  container escape.

This is a demonstrated architecture, not a shipped zero-knowledge sync
capability.
