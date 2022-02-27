#!/usr/bin/env python

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
    arg_parser.add_argument('--nanoaod', required=True, help='NanoAOD data')
    arg_parser.add_argument('--l1', required=True, help='L1 JEC file')
    arg_parser.add_argument('--l2', required=True, help='L2 JEC file')
    args = arg_parser.parse_args()

    events = NanoEventsFactory.from_root(args.nanoaod).events()
    jets = events.Jet
    num_jets = len(jets)
    print('Total number of jets:', num_jets)

    ext = extractor()
    ext.add_weight_sets([f'* * {args.l1}', f'* * {args.l2}'])
    ext.finalize()
    evaluator = ext.make_evaluator()

    l1_name = Path(args.l1).stem
    l2_name = Path(args.l2).stem
    jec_stack_names = [l1_name, l2_name]

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

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)

    start = time.time()
    corrected_jets = jet_factory.build(jets, lazy_cache=events_cache)
    end = time.time()
    inference_time = (end - start) / num_jets
    
    print('Inference time:', inference_time * 1000, 'ms per jet')
