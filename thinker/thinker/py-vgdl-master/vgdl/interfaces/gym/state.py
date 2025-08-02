from vgdl.state import StateObserver, KeyValueObservation
import math

from typing import Union, List, Dict


class AvatarOrientedObserver(StateObserver):

    def _get_distance(self, s1, s2):
        return math.hypot(s1.rect.x - s2.rect.x, s1.rect.y - s2.rect.y)


    def get_observation(self):
        avatars = self.game.get_avatars()
        assert avatars
        avatar = avatars[0]

        # Flatten avatar position
        avatar_pos = avatar.rect.topleft
        obs = KeyValueObservation()
        obs['position.x'] = avatar_pos[0]
        obs['position.y'] = avatar_pos[1]

        # Add speed if available
        if hasattr(avatar, 'speed'):
            obs['speed'] = avatar.speed
        else:
            obs['speed'] = 0

        # Flatten resources
        for i, r in enumerate(self.game.domain.notable_resources):
            obs[f'resource.{i}'] = avatar.resources.get(r, 0)

        # Flatten sprite distances
        for i, key in enumerate(self.game.sprite_registry.sprite_keys):
            dist = 100
            for s in self.game.get_sprites(key):
                dist = min(self._get_distance(avatar, s) / self.game.block_size, dist)
            obs[f'distance.{i}'] = dist
            
        return obs


class NotableSpritesObserver(StateObserver):
    """
    TODO: There is still a problem with games where the avatar
    transforms into a different type
    """
    def __init__(self, game, notable_sprites: Union[List, Dict] = None):
        super().__init__(game)
        self.notable_sprites = notable_sprites or game.sprite_registry.groups()


    def get_observation(self):
        state = []

        sprite_keys = list(self.notable_sprites)
        num_classes = len(sprite_keys)
        resource_types = self.game.domain.notable_resources

        for i, key in enumerate(sprite_keys):
            class_one_hot = [float(j==i) for j in range(num_classes)]

            # TODO this code is currently unsafe as getSprites does not
            # guarantee the same order for each call (Python < 3.6),
            # meaning observations will have inconsistent ordering of values
            for s in self.game.get_sprites(key):
                position = self._rect_to_pos(s.rect)
                if hasattr(s, 'orientation'):
                    orientation = [float(a) for a in s.orientation]
                else:
                    orientation = [0.0, 0.0]

                resources = [ float(s.resources[r]) for r in resource_types ]

                state += [
                    (s.id + '.position', position),
                    (s.id + '.orientation', orientation),
                    (s.id + '.class', class_one_hot),
                    (s.id + '.resources', resources),
                ]

        return KeyValueObservation(state)
