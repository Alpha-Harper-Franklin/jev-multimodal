# Transcript API integration check

`examples/transcript.json` contains authored English sentences, not captured audio.
The extractor preserved the two segment intervals and a source-content digest.
The hosted CLI submitted one Noul and one Choice together.

The first manual attempt failed with `transport_or_json_error`; its record is retained.
A second manual attempt produced `jev-response.json`. There is no automatic retry.
This checks JSON transcript import and the mixed typed API path, not ASR accuracy.
