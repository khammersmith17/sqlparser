# Overview
- This repo includes a basic sql parser, currently only supports select statements on a single table
- There is also a basic sqlite query engine, see sqlite/mod.rs for what is included in there
- so far, the modules in here support parsing a basic select statement, reading a sqlite schema, performing a full table scan on a sqlite table, querying using an index on a single table

## Plans
- This is more of an exploratory project for me to understand how query engines and databases work under the hood
- I plan to add more complex query logic in the sql parser
    - first GROUP BY, ORDER BY, and aggregation type commands
    - joins
- I also plan to add join logic into the query engine
- And then also try and implement some heuristics to make the query engine more robust
