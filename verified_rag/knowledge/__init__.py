"""Knowledge layer: recommendations, their slots, and links between them.

Every slot is a position in a chunk (chunk_id, start, end), never model text.
The model only points (Jev noul/choice) or the value is read by a fixed rule
(the WHO grade parenthetical). Links between documents come from the manifest;
links between recommendations across editions come from embedding similarity
of the stated text, and the grade comparison is exact string equality.
"""
