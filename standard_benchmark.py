from coffea.jetmet_tools import FactorizedJetCorrector
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.nanoevents import NanoEventsFactory
from coffea.lookup_tools import extractor
from pathlib import Path
import awkward as ak
import numpy as np
import argparse
import time


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('-n', '--nanoaod', required=True, help='NanoAOD data')
    arg_parser.add_argument('-j', '--jecfile', required=True, help='Jet energy corrections file')
    args = arg_parser.parse_args()

    events = NanoEventsFactory.from_root(args.nanoaod).events()
    jets = events.Jet
    print('# jets:', len(jets))

    ext = extractor()
    ext.add_weight_sets([f'* * {args.jecfile}'])
    ext.finalize()
    evaluator = ext.make_evaluator()

    jec_name = Path(args.jecfile).stem
    jec_stack_names = [jec_name]

    jec_inputs = {name: evaluator[name] for name in jec_stack_names}
    jec_stack = JECStack(jec_inputs)

    name_map = jec_stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['JetA'] = 'area'

    jets['pt_raw'] = (1 - jets['rawFactor']) * jets['pt']
    jets['mass_raw'] = (1 - jets['rawFactor']) * jets['mass']
    jets['pt_gen'] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
    jets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['Rho'] = 'rho'

    events_cache = events.caches[0]
    corrector = FactorizedJetCorrector(Summer19UL18_V5_MC_L1FastJet_AK4PFchs=evaluator[jec_name])

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)

    start = time.time()
    corrected_jets = jet_factory.build(jets, lazy_cache=events_cache)
    end = time.time()
    inference_time = (end - start) / len(jets)
    
    print('Inference time:', inference_time * 1000, 'ms')
