pub mod data_containers;
pub mod query_engine;
pub mod utils;

/* This module includes sqlite specific logic
* 1. includes utilites for reading sqlite varints, which differ slighlty from protobuf style
*    varints
* 2. There are a number of types that represent some on the on disk data structures in sqlite
*       - Database header, the first page in a sqlite database document
*       - sqlite page
*       - table schema
*       - index schema
*       - sqlite record
*       - supported sqlite serial types
*       - sqlite values
* 3. Additionally, there are some components for a query engine
*       - a basic query planner
*       - index traversal
*       - index page scans
*       - table page full scans
*       - table page scans for records in index
*       - btree traversal for full table scans and with index scan results
*       - searching within a btree page for records that satisfy and index condition
* */
