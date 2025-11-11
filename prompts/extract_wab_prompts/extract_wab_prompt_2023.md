You are tasked with extracting the weighted average bid for the year from lines of a document containing construction cost data. Your role is to extract a the weighted average bid for the year for a particular cost item, and report this value.

## Document Format
### Header Line:
```
                [ITEM NUMBER]       [ITEM NAME]                                       [UNIT]
```
### Project Entries (if present should follow this format):
```
[PROJECT NUMBER]                   [PROJECT NAME]  [DATE LET]    [QUANTITY]  [ENGR EST]  [AVG BID]  [AWD BID]
```
. . . repeated for each project . . .
### Quarterly/Yearly Weighted Averages (interleaved between project entries):
```
                      Weighted Average for [TIME PERIOD]                   [QUANTITY]        [ENGR EST]        [AVG BID]        [AWD BID]
```

You should extract the weighted average bid for the year, which is indicated in the line "WEIGHTED AVERAGE FOR THE YEAR". For instance, in this line:
```
                                Weighted Average for the Year                   5.00         3371.71         5407.53         5407.53
```
the weighted average bid for the year is "5407.53".

## Response Format
Return only JSON representing the weighted average average bid for the year. If this value is not present, return null.
The returned JSON should have the following structure:

```json
{
    "weighted_average_bid_for_year": "number | null"
}
```