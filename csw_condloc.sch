{	
	"BEGIN": {
		"sent": 
			"One [location.time] [subject.name] walked in to the [location.name]",
		"edge": {
			"location.latent.true": {"KNEW": 0.5, "TOLD": 0.5},
			"location.latent.false": {"KNEW": 0.5, "TOLD": 0.5}
		}
	},

	"KNEW": {
		"sent": 
			"[subject.pronoun] knew the '[drink.name]' was good here.",
		"edge":{
			"location.latent.true": {"AFTER": 0.8, "BEFORE": 0.2},
			"location.latent.false": {"AFTER": 0.2, "BEFORE": 0.8}
		}
	},
	"TOLD": {
		"sent": 
			"[subject.pronoun] was told the '[drink.name]' was good here.",
		"edge": {
			"location.latent.true": {"AFTER": 0.2, "BEFORE": 0.8},
			"location.latent.false": {"AFTER": 0.8, "BEFORE": 0.2}
		}
	},

	"AFTER": {
		"sent": 
			"After ordering, [subject.name] realized [realize.this]",
		"edge": {
			"location.latent.true": {"RADIO": 0.8, "TELE": 0.2},
			"location.latent.false": {"RADIO": 0.2, "TELE": 0.8}
		}
	},
	"BEFORE": {
		"sent": 
			"Before ordering, [subject.name] realized [realize.this]",
		"edge": {
			"location.latent.true": {"RADIO": 0.2, "TELE": 0.8},
			"location.latent.false": {"RADIO": 0.8, "TELE": 0.2}
		}
	},

	"RADIO": {
		"sent": 
			"Suddenly, the radio began '[broadcast.this]'",
		"edge": {
			"location.latent.true": {"PARENTS": 0.8, "SHAMAN": 0.2},
			"location.latent.false": {"PARENTS": 0.2, "SHAMAN": 0.8}
		}
	},
	"TELE": {
		"sent": 
			"Suddenly, the television began '[broadcast.this]'",
		"edge": {
			"location.latent.true": {"PARENTS": 0.2, "SHAMAN": 0.8},
			"location.latent.false": {"PARENTS": 0.8, "SHAMAN": 0.2}
		}
	},

	"PARENTS": {
		"sent": 
			"[subject.name] then remembered [subject.pospronoun] parents saying:",
		"edge": {
			"location.latent.true": {"STORY": 0.8, "NARRATIVE": 0.2},
			"location.latent.false": {"STORY": 0.2, "NARRATIVE": 0.8}
		}
	},
	"SHAMAN": {
		"sent": 
			"[subject.name] then remembered [subject.pospronoun] shaman-guru saying:",
		"edge": {
			"location.latent.true": {"STORY": 0.2, "NARRATIVE": 0.8},
			"location.latent.false": {"STORY": 0.8, "NARRATIVE": 0.2}
		}
	},

	"STORY": {
		"sent": 
			"'Our story is [saying.this]'",
		"edge": {
			"location.latent.true": {"TRANSPORT": 0.8, "TRANSCEND": 0.2},
			"location.latent.false": {"TRANSPORT": 0.2, "TRANSCEND": 0.8}
		}
	},
	"NARRATIVE": {
		"sent": 
			"'Our narrative is [saying.this]'",
		"edge": {
			"location.latent.true": {"TRANSPORT": 0.2, "TRANSCEND": 0.8},
			"location.latent.false": {"TRANSPORT": 0.8, "TRANSCEND": 0.2}
		}
	},

	"TRANSCEND": {
		"sent": 
			"That's when [subject.name] transcended.",
		"edge": {
			"location.latent.true": {"END": 1.0},
			"location.latent.false": {"END": 1.0}
		}
	},
	"TRANSPORT": {
		"sent": 
			"That's when [subject.name] was transported.",
		"edge": {
			"location.latent.true": {"END": 1.0},
			"location.latent.false": {"END": 1.0}
		}
	},

	"END": {
		"sent": 
			"~ FIN ~",
		"edge": {}
	}
}