Please extract all items from the provided project table, and return them in a JSON array. Each object in the array should contain the following fields:
- `item_number`: The item number from the project table
- `item_name`: The item name from the project table
- `unit`: The unit of measurement for the item
- `quantity`: The total amount of the item, taken from the project total  

Important notes:
- It is very important that the quantity be the project total for that item.
- Copy each field exactly as it appears in the project table, including spelling and capitalization.
- Output only the JSON array, *without* any additional text, code fences, or explanation.

Example output format:
[
    {
      "item_number": "203-00060",
      "item_name": "Embankment Material (Complete In Place)",
      "unit": "CY",
      "quantity": 962
    },
    {
      "item_number": "203-01597",
      "item_name": "Potholing",
      "unit": "HOUR",
      "quantity": 10
    },
    {
      "item_number": "206-00000",
      "item_name": "Structure Excavation",
      "unit": "CY",
      "quantity": 608
    },
    {
      "item_number": "206-00050",
      "item_name": "Structure Backfill (Special)",
      "unit": "CY",
      "quantity": 380
    }
    /* ... additional items ... */
]