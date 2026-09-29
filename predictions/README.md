# Sealed predictions

**Write-once.** Files in this directory are never modified after creation.
CI (`.github/workflows/integrity.yml`) fails the build if an existing file here
is changed or deleted.

One file per delivery day, `YYYY-MM-DD.json`, named for the **delivery** day,
not the day the forecast was made. Each contains:

- the seal timestamp (UTC), which must precede 12:00 Europe/Berlin on D-1;
- all three pre-registered calls for every hour of the local power day;
- `model_version` and a SHA-256 of the model source and frozen parameters;
- SHA-256 hashes of every input payload used to produce it.

A day whose job failed is recorded as `MISSED` and cannot be backfilled. A
prediction sealed after the auction closed is not a prediction.
