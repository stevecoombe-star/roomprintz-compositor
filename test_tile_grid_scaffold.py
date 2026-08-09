import unittest

from fastapi import HTTPException

from main import (
    AFC_SR1_TILE_GRID_SCAFFOLD_PROFILE,
    BASE_ROOMPRINTZ_INSTRUCTIONS,
    FLOORING_TILE_FRAGMENT,
    FLOORING_TILE_GRID_SCAFFOLD_FRAGMENT,
    TILE_GRID_SCAFFOLD_FLOORING_PRESET,
    VibodeStageRunRequest,
    _validate_stage2_research_scaffold_policy,
    build_stage2_surfaces_prompt_v1,
)


class TileGridScaffoldPromptTests(unittest.TestCase):
    def test_production_tile_prompt_is_unchanged_and_isolated(self):
        prompt = build_stage2_surfaces_prompt_v1(
            repair_damage=False,
            heavy_declutter=False,
            renovate_room=False,
            repaint_walls=False,
            flooring_preset="tile",
        )
        expected = "\n\n".join(
            [
                BASE_ROOMPRINTZ_INSTRUCTIONS.strip(),
                "You are given a single interior room photo. Edit this photo in-place for a surfaces/finishes pass.",
                FLOORING_TILE_FRAGMENT.strip(),
                (
                    "Output requirements:\n"
                    "- Return a single, high-quality edited image.\n"
                    "- The edit must look like a real photograph, not an illustration or painting.\n"
                    "- Do not alter the room's basic layout, window views, or camera angle."
                ),
            ]
        )
        self.assertEqual(prompt, expected)
        self.assertNotIn(FLOORING_TILE_GRID_SCAFFOLD_FRAGMENT.strip(), prompt)

    def test_research_scaffold_prompt_retains_load_bearing_analytical_requirements(self):
        prompt = build_stage2_surfaces_prompt_v1(
            repair_damage=False,
            heavy_declutter=False,
            renovate_room=False,
            repaint_walls=False,
            flooring_preset=TILE_GRID_SCAFFOLD_FLOORING_PRESET,
            research_profile=AFC_SR1_TILE_GRID_SCAFFOLD_PROFILE,
        )
        for required in (
            "square tiles",
            "large rectangular tiles are acceptable",
            "straight orthogonal grid",
            "grey / greyscale tiles",
            "clearly visible",
            "darker than the tiles",
            "low-reflection",
            "Do not use white or near-white tile",
            "Do not use diagonal installation",
            "herringbone",
            "chevron",
            "hexagonal",
            "Preserve the exact camera viewpoint",
            "wall-floor intersections",
            "modify flooring only",
        ):
            self.assertIn(required, prompt)
        self.assertNotIn(FLOORING_TILE_FRAGMENT.strip(), prompt)

    def test_scaffold_requires_its_explicit_stage_two_research_profile(self):
        valid = VibodeStageRunRequest(
            stage=2,
            flooringPreset=TILE_GRID_SCAFFOLD_FLOORING_PRESET,
            researchProfile=AFC_SR1_TILE_GRID_SCAFFOLD_PROFILE,
        )
        _validate_stage2_research_scaffold_policy(valid)

        invalid = (
            VibodeStageRunRequest(stage=2, flooringPreset=TILE_GRID_SCAFFOLD_FLOORING_PRESET),
            VibodeStageRunRequest(stage=2, flooringPreset="tile", researchProfile=AFC_SR1_TILE_GRID_SCAFFOLD_PROFILE),
            VibodeStageRunRequest(
                stage=1,
                flooringPreset=TILE_GRID_SCAFFOLD_FLOORING_PRESET,
                researchProfile=AFC_SR1_TILE_GRID_SCAFFOLD_PROFILE,
            ),
        )
        for request in invalid:
            with self.assertRaises(HTTPException) as raised:
                _validate_stage2_research_scaffold_policy(request)
            self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
