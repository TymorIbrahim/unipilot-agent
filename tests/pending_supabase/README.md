# Held back until the Supabase port

These five suites exercise the agent's DATA layer, which still speaks MongoDB
(`motor`, `compile_to_mongo`, collection-shaped source schemas). They are kept
verbatim rather than deleted because they encode the behaviour the Supabase
backend has to reproduce -- notably `test_predicate`'s operator semantics and
`test_reachability`'s guarantee that every advertised tool is reachable by a
route a model can actually walk.

Port them alongside `compile_to_sql`; a Supabase backend that cannot satisfy
these is not finished.
