You are tasked with extracting the weighted average bid for the year from lines of a document containing construction cost data. Your role is to extract a the weighted average bid for the year for a particular cost item, and report this value.

## Document Format
### Header Line:
```
=============== [ITEM NUMBER]      [ITEM NAME]                 [UNIT] ===============================================
```
### Project Entries (if present should follow this format):
```
[PROJECT NUMBER]                   [PROJECT NAME]  [DATE LET]    [QUANTITY]  [ENGR EST]  [AVG BID]  [AWD BID]
```
. . . repeated for each project . . .
### Quarterly/Yearly Weighted Averages (interleaved between project entries):
```
                      WEIGHTED AVERAGE FOR [TIME PERIOD]                   [QUANTITY]        [ENGR EST]        [AVG BID]        [AWD BID]
```

You should extract the weighted average bid for the year, which is indicated in the line "WEIGHTED AVERAGE FOR THE YEAR". For instance, in this line:
```
                      WEIGHTED AVERAGE FOR THE YEAR                   9.00        20000.00        56505.00        25000.00
```
the weighted average bid for the year is "56505.00".

## Response Format
Return only JSON representing the weighted average average bid for the year. If this value is not present, return null.
The returned JSON should have the following structure:

```json
{
    "weighted_average_bid_for_year": "number | null"
}
```