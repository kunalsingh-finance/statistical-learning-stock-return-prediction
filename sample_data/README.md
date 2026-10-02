# Historical public proxy sample

This existing sample contains characteristics and a same-month `mom1m` stand-in response called `RET`. It contains no verified CRSP return field. The corrected runner shifts that response to the next calendar month by security and excludes gaps/endpoints; use `--data-kind public_proxy` to label the run accurately.

These rows are a separate demonstration input. The checked-in `sample_outputs/` now come from generated synthetic data and do not describe this market-style sample. Feature publication lags, historical membership, delistings and the proxy's return accounting are unverified.
