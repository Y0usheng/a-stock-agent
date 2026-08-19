import unittest

from agent.prompt_registry import render_versioned_prompt, resolve_prompt_version


class PromptRegistryTests(unittest.TestCase):
    def test_versions_are_immutable_and_hash_the_rendered_prompt(self):
        v1, manifest1 = render_versioned_prompt("base", "manager-v1")
        v2, manifest2 = render_versioned_prompt("base", "manager-v2")
        self.assertEqual(v1, "base")
        self.assertNotEqual(v1, v2)
        self.assertNotEqual(manifest1["sha256"], manifest2["sha256"])
        self.assertEqual(manifest2["version"], "manager-v2")

    def test_unknown_version_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_prompt_version("manager-v999")


if __name__ == "__main__":
    unittest.main()
