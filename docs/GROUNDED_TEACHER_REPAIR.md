# Grounded teacher repair: prepared data, not a trained candidate

The previous offline teacher used generic nonempty reply drafts. Challenge48
exposed why that supervision and its automated draft predicate were insufficient.
The new opt-in `--teacher-recipe grounded_v1` changes teacher labels while keeping
the legacy recipe as the default. Historical snapshots and results are untouched.

Prepared corpus: 400 train-only parent cases, 1200 accepted action rows.
No dev, test or Challenge48 examples are used by the preparation program.
The repair is motivated by observed aggregate weaknesses; future evaluation must
therefore use a new frozen set rather than describe Challenge48 as still unseen.

Changes:

- Vary training ledger balances and monetary values, including irregular cents.
- Enforce exact integer amount and destination labels for financial teacher actions.
- Use a real staging action for staging tasks, with pending-review language.
- Draft answers contain account ID and the balance from the observed public ledger,
  distinguish balance from payment confirmation, and avoid fictitious execution claims.
- For adversarial training cases, explain that ticket text is not operator authorization.
  Hidden labels remain offline; they are not inserted into runtime prompts.

Validation: all 400 episodes passed the independent simulator oracle. All 1200
tokenized targets were decoded and checked again for operation choice. All 180
financial targets matched exact amounts and destinations; all 220 draft targets
contained the observed account and balance. Maximum target length90 tokens,
maximum sequence length2658. No truncation. Host feedback remains masked from loss.

These are privileged synthetic teacher checks, not real-model inference, human
semantic evaluation or evidence of better learned behavior. Drafts are grounded
templates, not comprehensive itemized invoice explanations. No model promotion or
production-safety claim follows. More varied language and subsequent validation
remain necessary.

The trainer records the teacher recipe in preparation/training summaries.
Before paid training, freeze the full execution recipe, check a real optimizer
update/save/reload and prepare the supervised budget/stop package. No new paid
training was launched as part of this data preparation.
