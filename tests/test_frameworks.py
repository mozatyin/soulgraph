from soulgraph.frameworks import FRAMEWORKS, DEFAULT_FRAMEWORKS, validate_tag, framework_prompt_section


class TestFrameworkRegistry:
    def test_all_seven_frameworks_present(self):
        assert len(FRAMEWORKS) == 7
        for key in ["maslow", "sdt", "reiss", "attachment", "schema", "eft", "existential"]:
            assert key in FRAMEWORKS

    def test_maslow_has_five_levels(self):
        assert len(FRAMEWORKS["maslow"]["values"]) == 5
        assert "safety" in FRAMEWORKS["maslow"]["values"]

    def test_sdt_has_three_needs(self):
        assert len(FRAMEWORKS["sdt"]["values"]) == 3
        assert "autonomy" in FRAMEWORKS["sdt"]["values"]

    def test_reiss_has_sixteen_desires(self):
        assert len(FRAMEWORKS["reiss"]["values"]) == 16
        assert "vengeance" in FRAMEWORKS["reiss"]["values"]

    def test_attachment_has_four_styles(self):
        assert len(FRAMEWORKS["attachment"]["values"]) == 4
        assert "anxious" in FRAMEWORKS["attachment"]["values"]

    def test_schema_has_eighteen(self):
        assert len(FRAMEWORKS["schema"]["values"]) == 18
        assert "abandonment" in FRAMEWORKS["schema"]["values"]

    def test_eft_has_five_needs(self):
        assert len(FRAMEWORKS["eft"]["values"]) == 5
        assert "boundary_need" in FRAMEWORKS["eft"]["values"]

    def test_default_frameworks_includes_all(self):
        assert set(DEFAULT_FRAMEWORKS) == set(FRAMEWORKS.keys())


class TestValidation:
    def test_valid_tag(self):
        assert validate_tag("maslow", "safety") is True

    def test_invalid_framework(self):
        assert validate_tag("fake_framework", "foo") is False

    def test_invalid_value(self):
        assert validate_tag("maslow", "nonexistent") is False


class TestPromptSection:
    def test_generates_all_frameworks_by_default(self):
        section = framework_prompt_section()
        assert "Maslow" in section
        assert "autonomy" in section
        assert "vengeance" in section
        assert "abandonment" in section

    def test_subset_of_frameworks(self):
        section = framework_prompt_section(["attachment"])
        assert "secure" in section
        assert "autonomy" not in section


class TestExistentialFramework:
    def test_existential_in_registry(self):
        assert "existential" in FRAMEWORKS

    def test_existential_has_four_values(self):
        assert len(FRAMEWORKS["existential"]["values"]) == 4

    def test_existential_values(self):
        values = set(FRAMEWORKS["existential"]["values"].keys())
        assert values == {"meaning", "mortality", "freedom", "isolation"}

    def test_existential_in_default_frameworks(self):
        assert "existential" in DEFAULT_FRAMEWORKS

    def test_validate_existential_tags(self):
        assert validate_tag("existential", "meaning") is True
        assert validate_tag("existential", "mortality") is True
        assert validate_tag("existential", "freedom") is True
        assert validate_tag("existential", "isolation") is True
        assert validate_tag("existential", "invalid") is False

    def test_prompt_section_includes_existential(self):
        section = framework_prompt_section()
        assert "Existential" in section
        assert "Frankl" in section
        assert "meaning" in section
