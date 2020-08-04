import torch
import streamlit as st
try:
    from edflow.util.edexplore import isimage, st_get_list_or_dict_item
except:
    def st_get_list_or_dict_item(ex, key, **kwargs):
        return ex[key], key
    isimage = lambda x: x

import autoencoders


@st.cache(allow_output_mutation=True)
def get_state(gpu, name="animals"):
    model = autoencoders.get_model(name)
    if gpu:
        model.cuda()
    state = {"model": model}
    return state


def reconstruction(ex, config):
    st.write("Options")

    name = st.selectbox(
        "Model Name",
        ["bigae_animals", "bigae_animalfaces"],
    )

    if torch.cuda.is_available():
        gpu = st.checkbox("gpu", value=True)

    state = get_state(name=name, gpu=gpu)
    model = state["model"]

    image, image_key = st_get_list_or_dict_item(ex, "image",
                                                description="input image",
                                                filter_fn=isimage)
    st.write("Input")
    st.image((image+1.0)/2.0)

    xin = torch.tensor(image)[None,...].transpose(3,2).transpose(2,1).float()
    if gpu:
        xin = xin.cuda()

    p = model.encode(xin)
    xout = model.decode(p.mode())

    xout = xout[0].transpose(0,1).transpose(1,2)
    xout = xout.detach().cpu().numpy()
    st.write("Reconstruction")
    st.image((xout+1.0)/2.0)


def sample(ex, config):
    name = st.selectbox(
        "Model Name",
        ["biggan_128", "biggan_256"],
    )
    class_conditional = True

    if torch.cuda.is_available():
        gpu = st.checkbox("gpu", value=True)

    select_cls = st.checkbox("select class", value=False)

    state = get_state(name=name, gpu=gpu)
    model = state["model"]

    z = ex["z"]
    if select_cls:
        cls = int(st.number_input("class", min_value=0, value=0))
    else:
        cls = ex["class"]

    zin = torch.tensor(z)[None,...]
    clsin = torch.tensor(cls)[None,...]
    if gpu:
        zin = zin.cuda()
        clsin = clsin.cuda()

    if class_conditional:
        xout = model.decode(zin, clsin)
    else:
        xout = model.decode(zin)

    xout = xout[0].transpose(0,1).transpose(1,2)
    xout = xout.detach().cpu().numpy()
    st.write("Sample")
    st.image((xout+1.0)/2.0)


@st.cache(allow_output_mutation=True)
def get_dset(cls, cfg):
    return cls(cfg)


if __name__ == "__main__":
    demo = st.sidebar.selectbox(
        "Demo",
        ["reconstruction", "sample"],
    )
    if st.sidebar.button("Reload Dataset"):
        st.caching.clear_cache()

    if demo=="reconstruction":
        from autoencoders.data import Folder
        cfg = {"Folder": {"folder":"assets", "size":128 }}
        dset = get_dset(Folder, cfg)
        dataidx = st.sidebar.slider("data index",0, len(dset)-1, 0)
        example = dset[dataidx]
        reconstruction(example, None)
    elif demo=="sample":
        from autoencoders.data import TestSamples
        n_test_samples = int(st.sidebar.number_input("n_test_samples", value=100))
        z_shape = int(st.sidebar.number_input("z_shape", value=120))
        n_classes = int(st.sidebar.number_input("n_classes", value=1000))
        truncation = float(st.sidebar.text_input("truncation", 0.0))
        cfg = {"BigGANData": {
            "n_test_samples": n_test_samples,
            "z_shape": (z_shape,),
            "n_classes": n_classes,
            "truncation": truncation,
        }}
        dset = get_dset(TestSamples, cfg)
        dataidx = st.sidebar.slider("data index",0, len(dset)-1, 0)
        example = dset[dataidx]
        sample(example, None)
