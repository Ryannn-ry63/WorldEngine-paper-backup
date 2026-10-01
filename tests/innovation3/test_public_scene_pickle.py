import unittest

from innovation3.scene_payload import single_scene_mapping


class PublicScenePickleTest(unittest.TestCase):
    def test_direct_scene_dictionary_is_normalized(self):
        scene = {'id': 'public-scene', 'map': 'us-nv-las-vegas-strip'}
        self.assertEqual(single_scene_mapping(scene, 'public-scene'), {'public-scene': scene})

    def test_combined_mapping_is_normalized(self):
        scene = {'id': 'public-scene'}
        self.assertEqual(single_scene_mapping({'public-scene': scene}, 'public-scene'),
                         {'public-scene': scene})

    def test_neighbouring_scene_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'does not contain requested scene'):
            single_scene_mapping({'id': 'other-scene'}, 'public-scene')
