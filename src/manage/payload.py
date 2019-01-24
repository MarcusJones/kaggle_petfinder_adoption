from pathlib import Path
import bz2
from jinja2 import Template
import logging
logging.getLogger().setLevel(logging.info)
#%% Collect all imported tools
# UTILS_DIR = Path('~/kaggle/kaggle_utils/kaggle_utils/').expanduser()
# TEST_LINES_OUT_DIR = Path('').cwd() / '..' / 'tempdel.txt'
# TEST_ZIP_OUT_DIR = Path('').cwd() / '..' / 'tempdel.bz2'

#%% TEMPLATES
PATH_TEMPLATE_HEAD = Path().cwd()/'src'/'manage'/'templates'/'head.py'
with PATH_TEMPLATE_HEAD.open() as fh:
    TEMPLATE_HEAD = Template(fh.read())

PATH_TEMPLATE_PAYLOADS= Path().cwd()/'src'/'manage'/'templates'/'payloads.py'
with PATH_TEMPLATE_PAYLOADS.open() as fh:
    TEMPLATE_PAYLOADS = Template(fh.read())

PATH_TEMPLATE_BASE= Path().cwd()/'src'/'manage'/'templates'/'base.py'
with PATH_TEMPLATE_BASE.open() as fh:
    TEMPLATE_BASE = Template(fh.read())

TEMPLATE_CELL = Template('payload["{{ key }}"]="{{ payload_chars }}"')

#%% Payloads
payloads = dict()
payloads['transformers'] = {
    'path': Path('~/kaggle/kaggle_utils/kaggle_utils/').expanduser() / 'transformers.py',
    'payload_chars' : None,
    'cell' : None,
}

#%%
def bz2_hex_string(path_input):
    assert path_input.exists(), "Can't find {}. Ensure PosixPath!".format(path_input)
    with path_input.open() as fh:
        ascii_lines = fh.read()
    zipped_lines = bz2.compress(bytes(ascii_lines, 'utf8'))
    hex_string = zipped_lines.hex()
    logging.info("Generated size {} hex string from {}".format(len(hex_string), path_input))
    return hex_string

# def wrap_cell(key, payload_chars):
#     cell_text =
#     return cell_text

for payload_key in payloads:
    # path_transformers = UTILS_DIR / 'transformers.py'
    payload = bz2_hex_string(payloads[payload_key]['path'])
    payloads[payload_key]['payload_chars'] = payload
    payloads[payload_key]['cell'] = TEMPLATE_CELL.render(key=payload_key, payload_chars=payloads[payload_key]['payload_chars'])


print( payloads[payload_key]['cell'] )

#%% Input files

assert path_transformers.exists()
with path_transformers.open() as fh:
    utils_lines = fh.readlines()
with path_transformers.open() as fh:
    utils_lines = fh.read()



bz2.decompress(payload)

payload_hex_str = '425a68393141592653590a663e1c000a9a5f8044507afff013ff7f5feebfeffffa600b2fba1b770fa1cba3deda3e5d0cd6adb5ada4ab262da7d1d541e18a694f44c8d26c907a4d3200d00007a8c9a06d4002114f1326a6d20c8000000000000001a68253d46111a341901a001a340000000126a449aa7b050d354f34a7a46806d4f50d1906100000044a29e854fca7a486c91809a3d427a69a46689a31300004122421a4c102649e4860a9ea78993694f694d34fd4d43d4c8193d1e00208aa2480aebf0ad9df6ec93c64ddb2c584ca4619647b24212ebc1bce1f47f9c8b79cc0c35612b30a44f091278c10c6834885a0d0eefad214bad579aa34c091ed55f1adad171328aaecd223ccb1402daa015125b329951bed718b87ddae245d184409b5a06ca0f3d9107dea9b9fc1487b2ab4e122c8c0244da9c981e08f4b1b1e5c37ffabfa80492570805d6d080d576beee83bf462a950fa59ec17434acdace9c357c3cee732cfaf9fd779a58750c2fd8379174cd5111d630db677ca971d9d939a198384520963741216a0f6639c4f880a481d5271deb596031d362c3d2be8cd35462ee334b335f5d79c34dec3bc19201376259105220711540354eb798b033688c0f99353b305a05505300e0e19b434bf4731f2a580c11209440351dae1ef6955cb35797336086b552cce365da6fb7bedacbaeb2bf0e73948785f8efc8edcff19dd7fa1ea384e4d9b605d012459b08fb450d9b95faa96e364e5652db6f829010fa6e80a2a0e916c449035e3102d8ab4f3672a20b95ab60e307895b0722ddb23118ca1ae52252ae25743945b73a09b537cc5ceb754ce65d7c2633ab650dcc777b587839143bfe69441e94f9304da501527db97030513d9eaf8e9786f2cc818b2f7a13f18818290d2d24409b00f625c92a0384467d67ab91c367a8e1fc3a8767095a793ba7e14b7f30bec72e3ca0bb42d7c1306e189f15119a2899fb4194638f7598133a7bec295910e255732244987fbad2c682440050949348a2491a6496e0b75c9e0e5e4bd9368f0ec85ae78e98c5d4d18653798b05a198bbe0c56219ee28aed99ca552b510e818d005d37887228b22d695a9a4658be59677dbb941f2215770cae32c0de22a91bf1707325f208bcb49ac2a1be88a92a808a215c7960e9a482290194522c403234283091d7ae68631a0896b9c8cb2b65b928c6f9a0c6de32415e9527be319d99cb7f5882739c70bcb52dd3a486a3da61a05172ac0b634b2b89b241bc680999e762bdd03d2ccc60ad29a291bf4c6cf47a0e5b8ec0702a401548e04f56c89d132a249b21613a5745f2861bd3019b6cff6e70b3ada8849224202122bb7cbc0f1b9590e6840885e28592419ab2ddec83e2797664af6fc8910ba5f341f120f11e2aeb644b4413a1424686021190e028370f8fa7a8c8388442d903154360e3c1809fe0b83833c0e31442f2f2dc255a65241157bddc9033e9e62d5bcfd2eb0fd30233f342f6ae4c77cc2622e418afc252e880f22f08496de5debbfe3d4ecf9afe5ae53496989dbdd1b44318db1a1fa403e754134b7c0e993bed17c63398d6510a05504608c1455145018360cad4788f6a54ce09c20f4ea6d03d725980be0d082495e232a519710e136dd85099f333a2c35a5860540f991bf1cc702e8ff8f9e924378da3efcc6d1fabc33b816913eb1410c4834dea49208ede262964143b64721eaa8791bb9a6677e3c1a3f5483c10595bc178ae932597434d871067815789d050448926b758833b791ac90642b960905b3f00217ff555ce9911d410bb148a86c4ece40f51a5d0ae2eb4d87d522ebd24d836a6c88c8b2982d2405050a815eb77ec78cbd61b8de4b499b17974d714c19365c11877d5ba0643a0c3b6b334475e9f8f98723a5135412f307ac38b513719229d1075b4652568dca8b3012b51b74346d70aec3b9385510dcf6081749491c97194ffb527cb859cddad5c75bfb3ba940aa68f64041ae3195e9841012c9346521b103a8a284bcca625e05c2c9092cc45e6a9ad357ebd8aa3a25f4acea7ba774db9aa66a018d7a966005d0aa96795d6ba736e71105e02d0b69c5d5e70c9c2eb35b6bccce2566d0c65a21806d19010e6a865f3a495c8a8c970977a1180b894e28906522e46004a6203dd83684191a3e13c35c4060bb33dda39041884a2d2e02ac90508b31a5158c158400889a1bf01748bc854235a908dbf74df3cbaca51401e12a84d31cb1abea73eec6d312152186b5b5a0f6c1905ea6967c63e4c032eda3b27a83b370b5c908e37764a19858bd5da8a46281c3a3be7b6371cc6612771085c4dc6c2935d650bfafae341c5c83ce54a03605634b581e6eae0a99a8116e016731f47ba4c4481734f707c1059619369057087fa2873f1375e52f62908d460ab42c726a873d7b0d2fdf060ebd1bd37d9b1b2e11146b9388688bcad2d984315b9a89dbbca904c39897bea40f4f3a207211620b2fbc848926808c61201012118c4181118ab8b601cf0bd945464844d637b9dc07e193fea82e99eb4cd519653981f0a48652e496446ccdf56817bc6a2500ca3433100e4c95b5049f7b2924df2f6795528e26b0dfacb4b96242a18ee3b941416c70d84562482c11605450834a9c55a94c3499824835a6152a5888681d6416227391729a0ee03f880ce6ea362665ef89b74448845ef7695e760a85db648928740250a9d44b2d8c85d65363fc06d894afe4e6d1bc1c39e02f6c8a06c1b0d4fe5c748c3753c56e2841f0997bc33a3239d8df11990751d42505eb81c7a264856ab587035181df1097a0eefd9a0b4d8ca6c90adbba1e0b26b553365554a1115b1e7688da0682aa523508a9c09a6990749a15d8a938cc308ce1ceb96e4e2503541c641109f4f65ac22f7e9ce739540390548251904a9026b820460c0b07da9119bae980b150882e2284ce6c3b0a71e7b0634c910594ba35a26e9060c97d62fb8c026344b2b90e624d04ca339a8868a40c06075fb187ddc7ab05ebf236815044923478bb25ac5c6a80aa560f91bc8b6b3825ef0aa8db6ce49d730b1cf240acb01c26d7a15ce0301b5a052820f7d57de81a2a4d405a5e94aef5b36a94c9ca53eae1da51b986233fded98dc334a5de906a442b1a5c0636bc9117229082d265b78686d9c984ee22e4a50219306345015e08ce517621c07ab3a534863c605165381896858378c24996b8434c2e80ef1e354749b27f899c054d046bcc9a983051cacbcddae458549080dc62d26db0de5cb65a1c17dac5ad423c539da1a8fe789e05c223ee46a368ca8152508f603da9014d7b833841beea678207c81ef3e60db420ee5a2d12bb8b450e1105505541a10c0cb3da4b84309c51ce184ddc9018c48b02237d284ce9738373fc5dc914e142402998f8700'
len(payload_hex_str)

clean_lines = list()
for line in utils_lines:
    out_line = line.replace('#%%', "#")
    out_line = out_line.replace('# %%', "#")
    clean_lines.append(out_line)

del utils_lines


head = """
"""
clean_lines.insert(0, "#%% TRANSFORMERS\ntransfomers_module = r'''\n")
clean_lines.append("'''\n")
clean_lines.append("#%%\n")
clean_lines.append("from pathlib import Path\n")
clean_lines.append("util_outpath = Path.cwd() / 'transformers.py'\n")
clean_lines.append("with util_outpath.open('w') as fh: fh.writelines(transfomers_module)\n")
clean_lines.append("print('*** Wrote {}'.format(util_outpath))")
clean_lines.append("#%%\n")

# %%

with open(TEST_LINES_OUT_DIR, 'w') as fh:
    fh.writelines(clean_lines)
# %%

len(payload)
with open(TEST_ZIP_OUT_DIR, 'wb') as fh:
    fh.write(payload)
logging.info("Wrote {:0.1f}kb to {}".format(len(payload)/1000, TEST_ZIP_OUT_DIR))