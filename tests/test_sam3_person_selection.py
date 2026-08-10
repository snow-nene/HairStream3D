import unittest

import numpy as np

from scripts.infer_2d.sam3_person_selection import (
    assign_hair_to_groups,
    deduplicate_masks,
    select_primary_person,
)


class Sam3PersonSelectionTest(unittest.TestCase):
    def test_centered_subject_beats_large_border_background(self):
        primary = np.zeros((100, 100), dtype=bool)
        primary[18:100, 25:75] = True
        background_union = np.ones((100, 100), dtype=bool)
        background_union[24:82, 30:70] = False
        hair = np.zeros((100, 100), dtype=bool)
        hair[12:38, 30:70] = True
        hair[2:12, 82:96] = True

        selected = select_primary_person([background_union, primary], hair)

        self.assertEqual(selected, 1)

    def test_overlapping_people_are_not_chain_merged(self):
        primary = np.zeros((80, 80), dtype=bool)
        primary[15:75, 20:50] = True
        background = np.zeros((80, 80), dtype=bool)
        background[5:55, 45:72] = True

        unique = deduplicate_masks([primary, background])

        self.assertEqual(len(unique), 2)
        self.assertFalse(np.array_equal(unique[0] | unique[1], unique[0]))

    def test_near_duplicate_prompt_masks_are_deduplicated(self):
        first = np.zeros((60, 60), dtype=bool)
        first[10:50, 15:45] = True
        second = first.copy()
        second[10, 15:20] = False

        unique = deduplicate_masks([first, second])

        self.assertEqual(len(unique), 1)

    def test_hair_assignment_ignores_single_pixel_contact(self):
        hair = np.zeros((80, 80), dtype=bool)
        hair[20:35, 45:60] = True
        primary = np.zeros((80, 80), dtype=bool)
        primary[34, 45] = True
        secondary = np.zeros((80, 80), dtype=bool)
        secondary[22:50, 42:65] = True

        assignment = assign_hair_to_groups([hair], [primary, secondary])

        self.assertEqual(assignment, [1])

    def test_distant_hair_is_left_unassigned(self):
        hair = np.zeros((100, 100), dtype=bool)
        hair[2:8, 2:8] = True
        person = np.zeros((100, 100), dtype=bool)
        person[55:95, 55:95] = True

        assignment = assign_hair_to_groups([hair], [person])

        self.assertEqual(assignment, [-1])


if __name__ == "__main__":
    unittest.main()
