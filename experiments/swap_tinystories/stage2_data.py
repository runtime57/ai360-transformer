"""Stage 2: hand-written descriptions, forbidden-word check, clean-model filter, dev/test split."""

import random
import re

import torch

from common import ART, BPE, DATA, Runner, jdump, jload, load_model
from stage1_capabilities import answer_logprobs

TEST_END = ' Every morning it said, "'
PROBE_ENDS = [" It liked to eat"] * 5 + [" It lived in a"] * 5

# For each animal: 10 probe descriptions (food/home endings) and 10 different test descriptions (sound ending).
DESCRIPTIONS = {
    "cat": {
        "probe": [
            "Once upon a time, there was a small pet that chased a little red light on the wall.",
            "There was a furry pet with a rough tongue and tiny sharp teeth.",
            "A little pet rubbed its head against Mom's legs when she came home.",
            "One day, a sneaky pet jumped onto the kitchen table and took a fish.",
            "There was a furry pet that knocked a cup off the table with its paw.",
            "Once upon a time, there was a soft pet that hid under the bed when it was scared.",
            "A small furry pet slept in a basket full of warm clothes.",
            "There was a pet with white socks on its paws and a fluffy tail.",
            "One night, a little pet sat on the roof and looked at the moon.",
            "Ben had a quiet pet that walked softly on its paws and never made a noise.",
        ],
        "test": [
            "Once upon a time, there was a little animal with soft fur and a long tail. It liked to drink milk and chase mice.",
            "There was once a small furry pet with long whiskers and pointy ears. It loved to sleep in a warm spot by the window.",
            "In a cozy house lived a fluffy pet with green eyes. It played with a ball of yarn all day.",
            "Lily had a soft little pet that licked its paws to stay clean. It liked to curl up on her lap.",
            "One day, a small furry animal climbed up a tall tree and could not get down. It had sharp claws and a long tail.",
            "Tom had a little pet with stripes and whiskers. It did not like water, and it ran away from the bath.",
            "There was a fluffy pet that sat on the fence at night. Its eyes shined in the dark, and it hunted mice.",
            "Once upon a time, a small pet with soft paws lived with an old lady. She gave it a bowl of milk every day.",
            "A little furry animal liked to scratch the sofa with its claws. It always landed on its feet when it jumped.",
            "There was a sleepy pet that napped in a sunny box. It had a pink nose and long whiskers.",
        ],
    },
    "dog": {
        "probe": [
            "Once upon a time, there was a pet that sat and gave its paw when Ben asked.",
            "There was a fast pet that caught a flying disc in its mouth.",
            "A spotted pet stuck its head out of the car window and felt the wind.",
            "One day, a pet with a red collar ran around and around after its own tail.",
            "There was a big pet that pulled a sled through the snow.",
            "Once upon a time, there was a small pet that shook water off its fur after a bath.",
            "A loyal pet waited by the gate every day for the boy to come home from school.",
            "There was a pet with a short tail and a collar with a shiny tag.",
            "One day, a smart pet helped the farmer bring the animals back from the hill.",
            "Max had a pet that panted with its tongue out on hot days.",
        ],
        "test": [
            "Once upon a time, there was a big furry animal that wagged its tail when it was happy. It loved to play fetch with a stick.",
            "Tim had a loyal pet that dug holes in the yard to hide its bones.",
            "There was a friendly pet with floppy ears and a wet nose. It ran to the door when Dad came home.",
            "One day, a playful pet chewed on Mom's shoe and hid it in the garden.",
            "In a small house lived a brave pet that guarded the yard at night. It did not like the mail man.",
            "A little furry pet liked to go for walks on a leash in the park.",
            "There was a happy pet that licked Sam's face and jumped on him. It had a long pink tongue.",
            "Once upon a time, a big brown pet chased the ball across the grass and brought it back.",
            "There was a pet that sniffed everything with its nose to find hidden treats. It loved to roll in the mud.",
            "Anna had a fluffy pet that slept in a little wooden house in the yard. It loved to chase cars.",
        ],
    },
    "cow": {
        "probe": [
            "Once upon a time, there was a big spotted animal that walked slowly through the green field.",
            "There was a large farm animal with a long rope tied around its neck.",
            "A big gentle animal stood near the fence and looked at the children with its big eyes.",
            "One day, a big farm animal with short horns lay down in the sun.",
            "There was a heavy animal whose milk was made into butter.",
            "Once upon a time, there was a big animal that the farmer brought in from the field every night.",
            "A large animal with brown spots and a white face walked up the hill.",
            "There was a big, calm animal with a wet nose and a long tongue.",
            "One day, a big animal broke the fence and walked into the farmer's garden.",
            "Ben visited a farm and saw a huge animal with big hooves and a heavy head.",
        ],
        "test": [
            "Once upon a time, there was a big animal on the farm that gave the farmer fresh milk.",
            "There was a large animal with black and white spots that ate grass in the field all day.",
            "In a green field lived a big gentle animal with a bell around its neck.",
            "One day, a big animal with two horns stood in the meadow and chewed slowly.",
            "There was a large farm animal that stood still in the barn while the farmer milked it.",
            "Lily saw a big brown animal with a long tail that swished the flies away.",
            "Once upon a time, a big slow animal walked to the pond to drink water with its friends from the herd.",
            "There was a gentle farm animal that rested in the shade and chewed its food again and again.",
            "A big animal with a pink nose and soft ears lived on Grandpa's farm. Its milk was used to make cheese.",
            "There was a large animal that ate hay in the barn. It had a big round belly and four strong legs.",
        ],
    },
    "duck": {
        "probe": [
            "Once upon a time, there was a bird that paddled with its orange feet.",
            "There was a little bird that dove under the water with its tail up in the air.",
            "A white farm bird walked in a funny way from side to side.",
            "One day, a bird shook the water off its wings after a swim in the pond.",
            "There was a soft bird that floated like a little boat on the lake.",
            "Once upon a time, there was a bird that slept with its head tucked under its wing.",
            "A brown bird hid among the tall reeds at the edge of the water.",
            "There was a bird with short legs that could swim, walk, and fly.",
            "One day, a little bird got lost and found its way back to the farm pond.",
            "Sam saw a bird with a curly tail feather swimming next to the boats.",
        ],
        "test": [
            "Once upon a time, there was a little bird with yellow feathers that loved to swim in the pond.",
            "There was a bird with webbed feet and a flat orange beak.",
            "One day, a small bird waddled to the lake with its babies walking behind it.",
            "In the park, a bird floated on the water and dipped its head to find food.",
            "Lily threw bread into the pond, and a hungry bird paddled over to get it.",
            "There was a white bird that waddled on the farm and splashed in a big puddle.",
            "Once upon a time, a little bird with green feathers on its head swam with its friends on the river.",
            "There was a bird that kept its feathers dry with oil and loved rainy days.",
            "A small bird flew south with its friends when winter came, and then came back to the pond in spring.",
            "There was a fluffy bird that laid eggs in a nest of grass near the water.",
        ],
    },
    "frog": {
        "probe": [
            "Once upon a time, there was a little animal that hopped into the pond with a big splash.",
            "There was a tiny green animal that hid under a big leaf.",
            "A small animal with spots on its back jumped across the wet stones.",
            "One day, a little animal puffed up its throat like a balloon.",
            "There was a small animal with bumpy skin and long legs.",
            "Once upon a time, a prince was turned into a little green animal by a witch.",
            "A tiny animal climbed up the side of a wet bucket with its sticky toes.",
            "There was a small green animal that slept in the mud all winter.",
            "One day, a little animal jumped onto Ben's hand and felt cold and wet.",
            "Mia caught a little green animal in a jar and then let it go by the water.",
        ],
        "test": [
            "Once upon a time, there was a little green animal that jumped from one lily pad to another.",
            "There was a small animal with a long sticky tongue that caught flies.",
            "One day, a little green animal sat on a rock by the pond and blinked its big eyes.",
            "In the pond lived a small slimy animal that could jump very high with its long back legs.",
            "There was a tiny animal that started its life as an egg in the water and grew legs later.",
            "Lily found a small green animal in the garden after the rain. It hopped away into the grass.",
            "Once upon a time, a little animal with smooth wet skin swam in the pond.",
            "There was a small animal that sang loudly at night by the pond with its friends.",
            "A little green animal sat very still and waited for a bug to fly by.",
            "There was a small animal with webbed feet and bulging eyes that lived near the stream.",
        ],
    },
    "lion": {
        "probe": [
            "Once upon a time, there was a big wild animal that was the strongest in the land.",
            "There was a hungry wild animal that hid in the grass and waited for its dinner.",
            "A big yellow animal with strong legs ran very fast across the dry land.",
            "One day, a wild animal with big claws climbed onto a high rock to look around.",
            "There was a big animal with a heavy head and golden eyes that led its family.",
            "Once upon a time, there was a big wild animal in a circus that jumped through a ring.",
            "A big tan animal had a scary face but was kind to a little mouse.",
            "There was a wild animal with round ears and a big nose that walked proudly.",
            "One day, a big wild animal got a thorn in its paw, and a boy helped it.",
            "Tom saw a big wild animal on TV that drank from a muddy river.",
        ],
        "test": [
            "Once upon a time, there was a big wild animal with a golden mane. It was the king of the jungle.",
            "There was a strong animal with sharp teeth that hunted in the tall grass.",
            "One day, a big yellow animal with a fluffy mane lay under a tree in the hot sun.",
            "In the jungle lived a brave big animal that all the other animals were afraid of.",
            "There was a large wild animal with big paws and a long tail with a tuft at the end.",
            "Once upon a time, a big golden animal lived with its family in the wild and slept most of the day.",
            "Lily saw a big wild animal at the zoo. It walked back and forth in its cage and showed its big teeth.",
            "There was a proud animal that wore a crown and ruled over the whole jungle.",
            "A big wild animal hunted zebras on the hot plains.",
            "There was a big strong animal from Africa with yellow fur and a loud voice.",
        ],
    },
}

# Names of every candidate animal, their young / related forms and the sound words: none may appear anywhere.
FORBIDDEN = {
    "cat": ["cat", "cats", "kitten", "kittens", "kitty", "kitties", "kit", "puss", "pussy", "tomcat", "feline", "meow", "mew", "purr", "purred", "purring"],
    "dog": ["dog", "dogs", "puppy", "puppies", "pup", "pups", "doggy", "doggie", "hound", "canine", "woof", "bark", "barked", "barking", "bow-wow", "growl"],
    "cow": ["cow", "cows", "calf", "calves", "bull", "bulls", "cattle", "ox", "oxen", "heifer", "moo", "mooed"],
    "duck": ["duck", "ducks", "duckling", "ducklings", "drake", "quack", "quacked"],
    "sheep": ["sheep", "lamb", "lambs", "ewe", "ram", "baa"],
    "pig": ["pig", "pigs", "piglet", "piglets", "hog", "swine", "oink"],
    "frog": ["frog", "frogs", "froggy", "tadpole", "tadpoles", "toad", "toads", "ribbit", "croak", "croaked"],
    "lion": ["lion", "lions", "lioness", "cub", "cubs", "roar", "roared", "roaring"],
    "owl": ["owl", "owls", "owlet", "hoot", "hooted"],
    "bee": ["bee", "bees", "buzz", "buzzed", "buzzing", "hive"],
    "horse": ["horse", "horses", "pony", "foal", "neigh"],
    "snake": ["snake", "snakes", "hiss", "hissed"],
}


def check_forbidden():
    bad = []
    pat = re.compile(r"\b(" + "|".join(sorted({w for ws in FORBIDDEN.values() for w in ws}, key=len, reverse=True)) + r")\b", re.I)
    for a, sets in DESCRIPTIONS.items():
        assert len(sets["probe"]) == 10 and len(sets["test"]) == 10, a
        for kind, descs in sets.items():
            for d in descs:
                m = pat.findall(d)
                if m:
                    bad.append((a, kind, d, m))
    all_desc = [d for s in DESCRIPTIONS.values() for ds in s.values() for d in ds]
    assert len(set(all_desc)) == len(all_desc), "duplicate descriptions"
    return bad


def main():
    torch.set_num_threads(22)
    bad = check_forbidden()
    for b in bad:
        print("FORBIDDEN:", b)
    assert not bad
    print("forbidden-word check: OK (0 hits over", sum(len(d) for s in DESCRIPTIONS.values() for d in s.values()), "descriptions)")

    s1 = jload(ART / "stage1.json")
    animals = s1["kept"]
    gold = {a: s1["results"][a]["gold"] for a in animals}
    cands = [gold[a] for a in animals]
    tok, R = BPE(), Runner(load_model())

    probe = [dict(animal=a, idx=i, desc=d, prompt=d + PROBE_ENDS[i]) for a in animals for i, d in enumerate(DESCRIPTIONS[a]["probe"])]
    tests, kept = [], []
    for a in animals:
        for i, d in enumerate(DESCRIPTIONS[a]["test"]):
            prompt = d + TEST_END
            lp = answer_logprobs(R, tok, prompt, cands)
            pred = animals[max(range(len(cands)), key=lambda j: lp[j])]
            rec = dict(animal=a, idx=i, desc=d, prompt=prompt, clean_pred=pred, logprobs=dict(zip(animals, lp)))
            tests.append(rec)
            if pred == a:
                kept.append(rec)

    rng = random.Random(0)
    dev, test = [], []
    for a in animals:
        items = [r for r in kept if r["animal"] == a]
        rng.shuffle(items)
        n_dev = round(0.3 * len(items))
        dev += items[:n_dev]
        test += items[n_dev:]

    jdump(dict(animals=animals, gold=gold, gold_first_token={a: tok.tok_str(tok.encode(gold[a])[0]) for a in animals},
               forbidden=FORBIDDEN, test_end=TEST_END, probe_ends=sorted(set(PROBE_ENDS))), DATA / "meta.json")
    jdump(probe, DATA / "probe.json")
    jdump(tests, DATA / "test_all.json")
    jdump(dev, DATA / "dev.json")
    jdump(test, DATA / "test.json")

    print("\n| animal | test prompts | clean-correct | dev | test | clean mistakes |")
    print("|---|---|---|---|---|---|")
    for a in animals:
        mis = [r["clean_pred"] for r in tests if r["animal"] == a and r["clean_pred"] != a]
        print(f"| {a} | 10 | {sum(r['animal'] == a for r in kept)} | {sum(r['animal'] == a for r in dev)} | "
              f"{sum(r['animal'] == a for r in test)} | {', '.join(mis) or '-'} |")
    print("total kept", len(kept), "dev", len(dev), "test", len(test))


if __name__ == "__main__":
    main()
