# City Hotel source dataset

The project uses City Hotel H2 only, with 79,330 bookings and 31 source
variables. Do not combine H1 and H2 for the required experiments.

## Location

The repository source-data location is:

`data/H2.csv`

The unchanged source CSV is included with the project.
Do not edit or resave it to perform cleaning. Apply transformations in code.

## Source identity

Expected SHA-256:

`6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4`

The frozen split and existing validation evidence refer to this exact file.
Different exports can contain equivalent values but have different byte
formatting. Do not replace the source without checking identity and ordering.

Check the file from the repository root:

```powershell
Get-FileHash -LiteralPath ".\data\H2.csv" -Algorithm SHA256
```