# Part 3 — A graph of exact manoeuvres as the planner's value *(planned)*

Part 3 of [Orbital Rendezvous](../../README.md). [Part 2](../2_oriented_port/README.md)
found that a sampling planner on the known dynamics docks through the port
195 times in 200 when it is given a good value beyond its horizon, written by
hand from the geometry, and only 100 when that value is learned from its own
flights; Go-Explore docks 198 times by searching for minutes per flight. This
part computes the value instead of writing or learning it, from the
structure of the problem: the Clohessy-Wiltshire dynamics are linear and
solved in closed form, so the exact manoeuvre between two points is known.

**Status: design, nothing built yet.**

## Design

```
 OFFLINE, once                                                ONLINE, every 10 s
 1. NODES  ──►  2. EDGES  ──►  3. VALUE  ──────────────────►  4. PLANNER
 points        exact CW        shortest way                    sampling MPC of Part 2,
 round the     transfers,      to the port,                    the graph's value beyond
 station       kept if legal   dynamic programming             its horizon
                                    │ (optional)
                                    └──►  5. a network that compresses the value
```

1. **Nodes.** Points round the station, denser near the port, a few hundred.
2. **Edges.** Between neighbouring nodes $`\mathbf{p}_i \to \mathbf{p}_j`$ and
   for a time of flight $`T`$, the departure velocity follows in closed form
   from the blocks of the state transition matrix,

   ```math
   \mathbf{v}_0^{+} = \Phi_{rv}^{-1}(T)\,\big(\mathbf{p}_j - \Phi_{rr}(T)\,\mathbf{p}_i\big),
   \qquad
   c_{ij} = |\Delta\mathbf{v}_1| + |\Delta\mathbf{v}_2| + \lambda\,T
   ```

   and an edge is kept only if its arc stays legal: outside the keep-out
   sphere, or inside it within the approach cone.
3. **Value.** Dynamic programming (Dijkstra) on the graph,
   $`V(\mathbf{p}_i) = \min_j \big[c_{ij} + V(\mathbf{p}_j)\big]`$ with
   $`V(\text{port}) = 0`$: the way round the sphere, and the side, come out of
   the shortest path; nobody writes them.
4. **Planner.** For a moving state, the exact transfer to the best neighbouring
   node plus its value, $`V(\mathbf{s}) = \min_j \big[c(\mathbf{s}\to\mathbf{p}_j) + V(\mathbf{p}_j)\big]`$,
   at the end of the horizon of the sampling MPC of Part 2.
5. **Network** (optional): $`V`$ learned from the graph, for speed and to
   generalise.

Choices: a polar grid denser near the port, times of flight of 200, 400 and
800 s, the cost of time $`\lambda`$ as in Step 17b; the network only if the
first four blocks work. First experiment: the 10 starts from behind the
station, then the 200 starts of Part 2.

## Relation to prior work

Blocks 1–3 follow an established line: sampling-based planning under CWH
dynamics with two-impulse steering between sample points, by Starek, Pavone
and co-authors [13]. A learned value at the end of an MPC horizon is also
known [14]. Not found in the literature searched so far: the graph's value as
the terminal cost of a sampling MPC that flies and replans, and the comparison
with model-free RL, learned values and Go-Explore on one task. This part is an
extension, presented as such.

The references continue the numbering of [Part 2](../2_oriented_port/README.md#references).

## License

MIT, Lorenzo Monti; see [LICENSE](../../LICENSE).
