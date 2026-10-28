import json
import unittest

from scripts.check_repository_secrets import protected_path, scan_content


class RepositorySecretTests(unittest.TestCase):
    def test_credentials_and_runtime_paths_are_protected_but_template_is_allowed(self):
        for path in (".env", ".env.production", ".codex/auth.json", "backup/auth.json", "chatgpt-auth.json",
                     "credentials.json", "keys/private.pem", "data/cache.db", "logs/live/input.json", ".ssh/id_ed25519"):
            self.assertTrue(protected_path(path), path)
        for path in (".env.example", "agent/subscription.py", "README.md"):
            self.assertFalse(protected_path(path), path)

    def test_api_github_and_jwt_credentials_are_detected_without_echoing_value(self):
        values = [b"sk-" + b"A" * 40, b"ghp_" + b"B" * 36,
                  b"eyJ" + b"C" * 12 + b"." + b"D" * 15 + b"." + b"E" * 15]
        for value in values:
            findings = scan_content("innocent.txt", b"first line\n" + value)
            self.assertTrue(findings)
            self.assertEqual(findings[0]["line"], 2)
            self.assertNotIn(value.decode(), json.dumps(findings))

    def test_literal_token_fields_and_private_keys_are_detected(self):
        content = json.dumps({"refresh_token": "x" * 36}).encode()
        self.assertEqual(scan_content("settings.json", content)[0]["rule"], "credential_value")
        marker = b"-----BEGIN " + b"PRIVATE KEY-----"
        self.assertEqual(scan_content("innocent.txt", marker)[0]["rule"], "private_key")

    def test_empty_example_and_code_that_references_environment_are_allowed(self):
        content = b'OPENAI_API_KEY=\n' + b'key = os.environ.get("OPENAI_API_KEY")\n'
        self.assertEqual(scan_content(".env.example", content), [])


if __name__ == "__main__":
    unittest.main()
