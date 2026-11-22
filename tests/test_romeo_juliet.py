import pytest
from model.inference import parse_screenplay_text
from model.content_advisory import compute_parents_guide
from model.emotion_detector import compute_character_emotions


def test_romeo_juliet_no_false_characters():
    script = """
Published: 1597 categorie fiction, drama, romance
Source: httpshakespeare.mit.edu
About Shakespeare: william shakespeare was an english poet and playwright...
Shakespeare: the greatest dramatist
children: susanna, and twins hamnet and judith...

ROMEO:
O, she doth teach the torches to burn bright!
It seems she hangs upon the cheek of night
Like a rich jewel in an Ethiope's ear;
Beauty too rich for use, for earth too dear!
Did my heart love till now? forswear it, sight!
For I ne'er saw true beauty till this night.
I love thee, I adore thee with all my soul.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.
My bounty is as boundless as the sea, my love as deep.

FRIAR LAURENCE:
There art thou happy: Tybalt would kill thee,
but thou slewest Tybalt; there art thou happy.
A pack of blessings light upon thy back.
By holy marriage: when, and where, and how
We met, we woo'd and made exchange of vow.

CAPULET:
Hang thee, young baggage! disobedient wretch!
I tell thee what: get thee to church o' Thursday,
Or never after look me in the face!
Out on her, hilding! Speak not, reply not!
Against to-morrow: my heart is wondrous light,
Since this same wayward girl is so reclaim'd.
Good night: get thee to bed, and rest.

LADY CAPULET:
Therefore use none: let Romeo hence in haste,
Else, when he is found, that hour is his last.

NURSE:
Shake, quoth the dove-house: 'twas no need, I trow,
To bid me trudge.

OFFICER:
Raise up the Montagues: some others search!

BENVOLIO:
Good morrow, cousin. What sadness lengthens Romeo's hours?
The day is young, but new struck nine.

TYBALT:
What, drawn, and talk of peace! I hate the word,
As I hate hell, all Montagues, and thee:
Have at thee, coward!
"""
    df = parse_screenplay_text(script, "Romeo_and_Juliet")
    characters = list(df["Character"].unique())

    # Ensure metadata headers are never characters
    assert "Published" not in characters
    assert "Source" not in characters
    assert "About Shakespeare" not in characters
    assert "Shakespeare" not in characters
    assert "children" not in characters

    # Ensure dialogue clauses with colons are never characters
    assert "There art thou happy" not in characters
    assert "Depart again" not in characters
    assert "I fear it is" not in characters
    assert "Against to-morrow" not in characters
    assert "By holy marriage" not in characters
    assert "In dear employment" not in characters
    assert "Farewell" not in characters
    assert "Good night" not in characters
    assert "Therefore use none" not in characters
    assert "Shake quoth the dove-house" not in characters
    assert "Raise up the Montagues" not in characters

    # Ensure real characters are captured
    assert "ROMEO" in characters
    assert "JULIET" in characters
    assert "FRIAR LAURENCE" in characters
    assert "CAPULET" in characters
    assert "LADY CAPULET" in characters
    assert "NURSE" in characters
    assert "OFFICER" in characters
    assert "BENVOLIO" in characters
    assert "TYBALT" in characters

    # Check emotions: Realistic emotional differentiation
    em = compute_character_emotions(df)
    assert em["ROMEO"]["dominant_emotion"] == "joy"  # Love / Romance
    assert em["JULIET"]["dominant_emotion"] == "joy"  # Love / Romance
    assert em["TYBALT"]["dominant_emotion"] == "anger"  # Hatred / Hostility
    assert em["BENVOLIO"]["dominant_emotion"] in ("neutral", "sadness")  # Melancholy inquiry, NOT fake joy!
    assert em["CAPULET"]["dominant_emotion"] in ("neutral", "anger")

    # Check certification: should be U/A, NOT 'A'
    pg = compute_parents_guide(df)
    assert pg["cbfc_code"] in ("U/A 7+", "U/A 13+", "U/A 16+")
    assert pg["cbfc_code"] != "A"
