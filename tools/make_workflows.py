"""Generates the three example workflows (interface format + API format) in example_workflows/.

python tools/make_workflows.py   (no dependency)
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "example_workflows"
UNET, CLIP, VAE = ("p-h3-image_int8_convrot.safetensors", "p-h3-image_text_encoder_int8_convrot.safetensors",
                   "p-h3-image_vae_bf16.safetensors")
SIGMAS = "1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0"
W, H, W4, H4 = 768, 1344, 1536, 2688
PROMPT = ("A woman in a red jacket dancing in a sunlit street, one arm raised high, the other arm bent, hips swaying, "
          "full body, natural light, 35mm photo")
HANDS = ("Underwater photograph of a female scuba diver facing the camera, framed from the waist up, in clear tropical blue water "
         "above a coral reef, sun rays from the surface, small bubbles rising. She wears a black wetsuit, a diving mask and a regulator "
         "in her mouth. Both hands are raised at shoulder height, palms toward the camera: her right hand (on the left of the image) gives "
         "a thumbs-up, fist closed with only the thumb raised; her left hand (on the right of the image) makes the OK sign, the tip of the "
         "index finger touching the tip of the thumb to form a circle, the other three fingers raised and spread. Natural, unretouched "
         "underwater photograph, true-to-life colours, five fingers on each hand.")
MQ = ("Edit <Picture 1>. The two white mannequins with grey outlines in <Picture 1> are <Subject 1>. The target image keeps the "
      "composition of <Picture 1> and turns <Subject 1> into real people in a photorealistic photograph. <Subject 1> is partially "
      "preserved; the body pose, the angle of every limb, the camera viewpoint, the framing and the size and position of each figure "
      "remain, while their surface becomes real people with clothes. The grey backdrop is replaced by the real scene; no mannequin "
      "texture, outline or grey background remains. Acrobatic duo on a sunny tropical beach: a man in black swim trunks holds a wide "
      "squat and lifts his partner with one raised arm; she, in a red swimsuit, does a one-arm handstand on his hand, legs in a split. "
      "Bright daylight, sports photography.")

# widget names per node type, in the interface order (they become the API inputs)
WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"], "MiniMaxH3SigmaShift": ["shift_video", "shift_audio"],
    "CLIPLoader": ["clip_name", "type", "device"], "VAELoader": ["vae_name"], "LoadImage": ["image", "_upload"],
    "PH3PrepareControlImage": ["width", "height", "fit", "background", "invert"],
    "PH3ImageConditioning": ["prompt", "width", "height"],
    "PH3ControlSimple": ["control_type", "intensity", "auto_download", "patch_name"],
    "ImageScaleToTotalPixels": ["upscale_method", "megapixels", "resolution_steps"],
    "RandomNoise": ["noise_seed", "_cag"], "KSamplerSelect": ["sampler_name"], "ManualSigmas": ["sigmas"],
    "SaveImage": ["filename_prefix"], "Note": ["text"],
}
OUTS = {"UNETLoader": ["MODEL"], "MiniMaxH3SigmaShift": ["MODEL"], "CLIPLoader": ["CLIP"], "VAELoader": ["VAE"],
        "LoadImage": ["IMAGE", "MASK"], "PH3PrepareControlImage": ["IMAGE"], "ImageScaleToTotalPixels": ["IMAGE"],
        "PH3ImageConditioning": ["CONDITIONING", "LATENT"], "PH3ControlSimple": ["MODEL", "LATENT"],
        "BasicGuider": ["GUIDER"], "RandomNoise": ["NOISE"], "KSamplerSelect": ["SAMPLER"], "ManualSigmas": ["SIGMAS"],
        "SamplerCustomAdvanced": ["LATENT", "LATENT"], "PH3ControlFirstFrame": ["LATENT"], "VAEDecode": ["IMAGE"],
        "SaveImage": [], "Note": []}
TYPES = {"model": "MODEL", "clip": "CLIP", "vae": "VAE", "image": "IMAGE", "conditioning": "CONDITIONING", "latent": "LATENT",
         "control_image": "IMAGE", "images.image_1": "IMAGE", "guider": "GUIDER", "noise": "NOISE", "sampler": "SAMPLER",
         "sigmas": "SIGMAS", "latent_image": "LATENT", "samples": "LATENT", "images": "IMAGE"}


class G:
    """Tiny graph builder: add(type, pos, widgets, links, title) -> id."""
    def __init__(self):
        self.nodes = []

    def add(self, typ, pos, widgets=(), links=None, title=None, size=(300, 120)):
        nid = len(self.nodes) + 1
        self.nodes.append(dict(id=nid, type=typ, pos=pos, widgets=list(widgets), links=links or {}, title=title, size=size))
        return nid

    def build(self):
        links, ui_nodes, api = [], [], {}
        for n in self.nodes:
            for name, (o, slot) in n["links"].items():
                links.append([len(links) + 1, o, slot, n["id"], name, TYPES[name]])
        for order, n in enumerate(self.nodes):
            mine = [l for l in links if l[3] == n["id"]]
            node = {"id": n["id"], "type": n["type"], "pos": n["pos"], "size": list(n["size"]), "flags": {}, "order": order,
                    "mode": 0, "inputs": [{"name": l[4], "type": l[5], "link": l[0]} for l in mine],
                    "outputs": [{"name": t, "type": t, "links": [l[0] for l in links if l[1] == n["id"] and l[2] == k],
                                 "slot_index": k} for k, t in enumerate(OUTS[n["type"]])],
                    "properties": {"Node name for S&R": n["type"]}, "widgets_values": n["widgets"]}
            if n["title"]:
                node["title"] = n["title"]
            if n["type"] == "Note":
                node["color"], node["bgcolor"] = "#432", "#653"
            ui_nodes.append(node)
            if n["type"] != "Note":
                inputs = {name: [str(o), s] for name, (o, s) in n["links"].items()}
                inputs.update({k: v for k, v in zip(WIDGETS.get(n["type"], []), n["widgets"]) if not k.startswith("_")})
                api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs, "_meta": {"title": n["title"] or n["type"]}}
        ui_links = [[l[0], l[1], l[2], l[3], next(i for i, x in enumerate(ui_nodes[l[3] - 1]["inputs"]) if x["link"] == l[0]), l[5]]
                    for l in links]
        ui = {"id": "8f0d1c9a-0f3a-4c52-9b3e-0a1b2c3d4e5f", "revision": 0, "last_node_id": len(self.nodes),
              "last_link_id": len(links), "nodes": ui_nodes, "links": ui_links, "groups": [], "config": {},
              "extra": {"ds": {"scale": 0.6, "offset": [0, 0]}}, "version": 0.4}
        return ui, api


def base(g, prompt, w, h, image, x=40, ref=None):
    """Model + text + image loading + conditioning. Returns ids."""
    unet = g.add("UNETLoader", [x, 40], [UNET, "default"], title="P-H3-Image model")
    shift = g.add("MiniMaxH3SigmaShift", [x + 340, 40], [1.0, 1.0], {"model": (unet, 0)})
    clip = g.add("CLIPLoader", [x, 170], [CLIP, "minimax", "default"])
    vae = g.add("VAELoader", [x, 310], [VAE])
    return unet, shift, clip, vae


def sampler_block(g, guider, latent, x, y):
    noise = g.add("RandomNoise", [x, y], [42, "randomize"])
    ks = g.add("KSamplerSelect", [x, y + 100], ["euler"])
    sg = g.add("ManualSigmas", [x, y + 180], [SIGMAS], size=(300, 90))
    sca = g.add("SamplerCustomAdvanced", [x + 330, y], [], {"noise": (noise, 0), "guider": (guider, 0), "sampler": (ks, 0),
                                                           "sigmas": (sg, 0), "latent_image": latent})
    return noise, ks, sg, sca


def control_stage(g, image_name, prompt, note, w, h, control_loader=None):
    unet, shift, clip, vae = base(g, prompt, w, h, image_name)
    note_id = g.add("Note", [40, 760], [note], title="Read me", size=(420, 220))
    load = g.add("LoadImage", [40, 450], [image_name, "image"], title="Control image (depth / pose / edges)", size=(300, 300))
    prep = g.add("PH3PrepareControlImage", [380, 450], [w, h, "fit", "black", False], {"image": (load, 0)})
    cond = g.add("PH3ImageConditioning", [380, 170], [prompt, w, h], {"clip": (clip, 0), "vae": (vae, 0)}, size=(360, 220))
    ctl = g.add("PH3ControlSimple", [780, 40], ["auto", "Normal (Normale) 1.2 · 50 %", False,
                                                 "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"],
                {"model": (shift, 0), "vae": (vae, 0), "control_image": (prep, 0), "latent": (cond, 1)}, size=(340, 200))
    guider = g.add("BasicGuider", [1150, 40], [], {"model": (ctl, 0), "conditioning": (cond, 0)})
    _, _, _, sca = sampler_block(g, guider, (ctl, 1), 1150, 140)
    first = g.add("PH3ControlFirstFrame", [1500, 40], [], {"latent": (sca, 0)})
    dec = g.add("VAEDecode", [1500, 120], [], {"samples": (first, 0), "vae": (vae, 0)})
    return dict(shift=shift, clip=clip, vae=vae, dec=dec, noise=sca - 3, ks=sca - 2, sg=sca - 1)


NOTE_BASIC = ("1) Put the MiniMax H3 Fun ControlNet v1 patch (2.3 GB) in models/model_patches, or tick auto_download on the "
              "control node. 2) Pick a control image (examples: custom_nodes/comfyui-ph3-control/example_workflows/control_images; "
              "copy one into ComfyUI/input). 3) DESCRIBE THE POSE in the prompt. 4) Run. Experimental. Intensity: Light / Normal / Strict.")
NOTE_4 = ("High quality: the control runs directly at ~4 MP (1536x2688), no upscaling pass. The control image is an OpenPose "
          "skeleton WITH hands (21 points per hand): the skeleton places the hands, the prompt names the gesture (thumbs-up, OK...). "
          "Copy control_images/pose_hands_diver_9x16.png into ComfyUI/input. About 45 s on an RTX 3090. Do not go to 8 MP: the pose drifts. "
          "Tick auto_download or put the v1 patch in models/model_patches. Experimental.")
NOTE_MQ = ("No ControlNet, no patch. A white mannequin with grey outlines is Picture 1 and the prompt uses the official H3 grammar "
           "(<Picture 1>, <Subject 1>, 'partially preserved'). Draw the mannequin with any pose editor (a ready example is provided), "
           "put it in ComfyUI/input, change the scene at the end of the prompt. Faster than the ControlNet, a bit less exact. Experimental.")


def basic():
    g = G()
    control_stage(g, "depth_dance_9x16.png", PROMPT, NOTE_BASIC, W, H)
    save_src = [n for n in g.nodes if n["type"] == "VAEDecode"][0]["id"]
    g.add("SaveImage", [1500, 200], ["ph3_pose_control"], {"images": (save_src, 0)}, size=(320, 340))
    return g.build()


def hq4():
    g = G()
    s1 = control_stage(g, "pose_hands_diver_9x16.png", HANDS, NOTE_4, W4, H4)
    g.add("SaveImage", [1500, 200], ["ph3_pose_control_4mp"], {"images": (s1["dec"], 0)}, size=(320, 340))
    return g.build()


def mannequin():
    g = G()
    unet, shift, clip, vae = base(g, MQ, W, H, None)
    g.add("Note", [40, 760], [NOTE_MQ], title="Read me", size=(420, 220))
    load = g.add("LoadImage", [40, 450], ["mannequin_duo_9x16.png", "image"], title="Mannequin (Picture 1)", size=(300, 300))
    ref = g.add("ImageScaleToTotalPixels", [380, 450], ["lanczos", 1.0, 32], {"image": (load, 0)})
    cond = g.add("PH3ImageConditioning", [380, 170], [MQ, W, H], {"clip": (clip, 0), "vae": (vae, 0), "images.image_1": (ref, 0)}, size=(360, 220))
    guider = g.add("BasicGuider", [780, 40], [], {"model": (shift, 0), "conditioning": (cond, 0)})
    _, _, _, sca = sampler_block(g, guider, (cond, 1), 780, 140)
    dec = g.add("VAEDecode", [1130, 40], [], {"samples": (sca, 0), "vae": (vae, 0)})
    g.add("SaveImage", [1130, 120], ["ph3_mannequin_reference"], {"images": (dec, 0)}, size=(320, 340))
    return g.build()


if __name__ == "__main__":
    for name, fn in (("ph3_pose_control_basic", basic), ("ph3_pose_control_4mp_hands", hq4), ("ph3_mannequin_reference", mannequin)):
        ui, api = fn()
        (OUT / f"{name}.json").write_text(json.dumps(ui, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (OUT / f"{name}_api.json").write_text(json.dumps(api, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(name, len(ui["nodes"]), "nodes", len(ui["links"]), "links")
