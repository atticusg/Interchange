class Location:
    """A helper class to manage parsing of indices and slices"""

    def __init__(self, loc=None):
        self._loc = Location.process(loc) if loc else None

    def __getitem__(self, item):
        return item

    @classmethod
    def process(cls, x):
        if isinstance(x, int) or isinstance(x, list) or isinstance(x, tuple) \
                or isinstance(x, slice) or x is Ellipsis:
            return x
        elif isinstance(x, str):
            return Location.parse_str(x)

    @classmethod
    def parse_str(cls, s):
        if "," not in s:
            return Location.parse_dim(s.strip())
        else:
            return tuple(Location.parse_dim(x.strip()) for x in s.split(","))

    @classmethod
    def parse_dim(cls, s):
        s = s.strip("[]")
        return Ellipsis if s == "..." \
            else True if s == "True" \
            else False if s == "False" \
            else Location.str_to_slice(s) if ":" in s \
            else int(s)

    @classmethod
    def str_to_slice(cls, s):
        return slice(
            *map(lambda x: int(x.strip()) if x.strip() else None, s.split(':')))

    @classmethod
    def slice_to_str(cls, s):
        return ':'.join(x if x != "None" else "" for x in str(s).strip("slice").strip("()").split(", "))
