class WrappedProteinChain:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        seq = getattr(self, "sequence", None)
        return len(seq) if seq is not None else 0
