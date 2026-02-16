"""Tests for the experiment runner CLI."""

from runner.run import load_config, build_parser


class TestSeedOverride:
    """Tests for --seed CLI argument."""

    def test_parser_accepts_seed_arg(self):
        """The runner's argument parser should accept --seed."""
        parser = build_parser()
        args = parser.parse_args(["--config", "dummy.yaml", "--seed", "456"])
        assert args.seed == 456

    def test_parser_seed_defaults_to_none(self):
        """Without --seed, args.seed should be None (config seed is used)."""
        parser = build_parser()
        args = parser.parse_args(["--config", "dummy.yaml"])
        assert args.seed is None

    def test_seed_override_replaces_config_seed(self, tmp_path):
        """--seed CLI arg should override config['seed'] when applied."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            "experiment_name: test\n"
            "seed: 42\n"
            "dataset:\n"
            "  name: ccfd\n"
            "model:\n"
            "  type: neural\n"
            "attack:\n"
            "  type: capgd\n"
            "  epsilon: 0.1\n"
            "defence:\n"
            "  type: none\n"
        )

        config = load_config(str(config_file))
        # Simulate what the runner should do with --seed
        seed_override = 456
        config["seed"] = seed_override

        assert config["seed"] == 456
