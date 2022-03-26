import torch

class Room:
    def __init__(self, max_paths, embedding):
        self.max_paths = max_paths
        self.embedding = embedding
        self.paths = []

    def populate(self, root):
        embedding_size = len(self.embedding)
        for _ in range(torch.randint(1, self.max_paths, ())):
            mask = torch.rand(embedding_size) < 2 / embedding_size
            embedding = self.embedding + mask * torch.normal(0, 1, size=(embedding_size,))
            self.paths.append(Room(max_paths=self.max_paths, embedding=embedding))

class CaveEnvironment:
    def __init__(self, max_paths, embedding_size):
        self.root = Room(max_paths=max_paths, embedding=torch.zeros(embedding_size))
        self.root.populate(self.root)
        self.current_room = self.root
        self.rooms = [self.root]

    def step(self, action):
        if action < len(self.current_room.paths):
            self.current_room = self.current_room.paths[action]
            if not self.current_room.paths:
                self.current_room.populate(self.root)
                self.rooms.append(self.current_room)
        return self.current_room.embedding

    def reset(self):
        self.current_room = self.root
        return self.current_room.embedding
