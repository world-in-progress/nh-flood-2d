import fastdb4py as fdb

@fdb.feature
class IndexLike:
    index: fdb.U32

@fdb.feature
class U8Value:
    value: fdb.U8

@fdb.feature
class F32Value:
    value: fdb.F32

@fdb.feature
class Ne:
    index: fdb.U32
    x: fdb.F32
    y: fdb.F32
    z: fdb.F32
    l_side_num: fdb.U32
    r_side_num: fdb.U32
    b_side_num: fdb.U32
    t_side_num: fdb.U32
    type: fdb.U8

@fdb.feature
class SideTopoInfo:
    """
    Used to store side topology information for NsData
    A side has 5 topology info:
    0: orient (1: horizontal, 2: vertical)
    1: left or bottom Ne index (0: no Ne)
    2: right or top Ne index (0: no Ne)
    """
    info: fdb.U32

@fdb.feature
class Ns:
    index: fdb.U32
    length: fdb.F32
    x: fdb.F32
    y: fdb.F32
    z: fdb.F32
    attr: fdb.U8

@fdb.feature
class Node:
    index: fdb.U32
    name: fdb.STR
    x: fdb.F32
    y: fdb.F32
    is_outfall: fdb.BOOL

@fdb.feature
class PipeTopo:
    """
    CSR data array for pipe-node to 2D-element topology.

    Indexing contract:
      - Node table rows are 0-based (0 .. n_nodes-1).
      - topo_ptr (stored as IndexLike['topo_ptr']) is a 0-based CSR offset array,
        length = n_nodes + 1.  Node i owns topo_ei[ topo_ptr[i] : topo_ptr[i+1] ].
      - ei values are 1-based element indices (0 = invalid sentinel, same as repo-wide convention).
      - PipeTopo stores ONLY secondary/weak-related cells; primary_ei is stored separately
        in IndexLike['node_primary_ei'] and must NOT appear in this table.
    """
    ei: fdb.U32
