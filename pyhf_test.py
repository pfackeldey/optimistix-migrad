import os
import time

# for iminuit fit
import pyhf
import cabinetry
import iminuit

# for optimistix fit
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

from migrad_optimistix import Migrad


pyhf.set_backend("jax")  # make sure pyhf uses jax under-the-hood
jax.config.update("jax_enable_x64", True)


TOLERANCE = 1e-3


def prepare_pyhf(asimov=True):
    import json
    from pyhf.contrib.utils import download

    if not os.path.exists("bottom-squarks.json"):
        download(
            "https://www.hepdata.net/record/resource/1935437?view=true",
            "bottom-squarks",
        )
        ws = pyhf.Workspace(json.load(open("bottom-squarks/RegionC/BkgOnly.json")))
        patchset = pyhf.PatchSet(
            json.load(open("bottom-squarks/RegionC/patchset.json"))
        )
        ws = patchset.apply(ws, "sbottom_600_280_150")

        cabinetry.workspace.save(ws, "bottom-squarks.json")

    ws = cabinetry.workspace.load("bottom-squarks.json")
    model, data = cabinetry.model_utils.model_and_data(ws)

    init_pars = model.config.suggested_init()

    if asimov:
        pdf = model.make_pdf(jnp.array(init_pars))
        return model, pdf.sample((1,))[0], init_pars
    else:
        return model, data, init_pars


def twice_nll_init_pars_data(model, init_pars, data) -> jax.Array:
    def _twice_nll(pars_, data_):
        return -2 * model.logpdf(pars_, data_)[0]

    return eqx.filter_jit(_twice_nll), jnp.array(init_pars), jnp.array(data)


def timeit(fun):
    tic = time.monotonic()
    x = fun()
    toc = time.monotonic()
    return toc - tic, x


def prepare_iminuit(nll, init_pars, data):
    def likelihood(pars):
        return nll(pars, data)

    minuit = iminuit.Minuit(likelihood, init_pars, grad=eqx.filter_grad(likelihood))

    minuit.strategy = 1
    minuit.print_level = 0
    minuit.tol = TOLERANCE

    def fit():
        minuit.migrad(ncall=100_000, use_simplex=False)
        bestfit = jnp.array(minuit.values)
        minuit.reset()
        return bestfit

    return fit


def prepare_optimistix_migrad(nll, init_pars, data):
    def likelihood(pars, data):
        return nll(pars, data)

    # SIMPLEX
    # solver = optx.NelderMead(rtol=0.001, atol=1e-5)

    # MIGRAD
    solver = Migrad(
        rtol=TOLERANCE * iminuit.Minuit.LIKELIHOOD,
        atol=1e-5,  # ignored when using edm
        use_inverse=True,
        # verbose=frozenset({"edm"}),
    )

    # BFGS
    # solver = optx.BFGS(rtol=1e-3, atol=1e-5)

    def fit():
        bestfit = optx.minimise(
            likelihood,
            solver,
            init_pars,
            has_aux=False,
            args=data,
            options={},
            max_steps=100_000,
            throw=True,
        ).value
        return bestfit

    return fit


if __name__ == "__main__":
    model, data, init_pars = prepare_pyhf(asimov=False)

    nll, init_pars, data = twice_nll_init_pars_data(model, init_pars, data)

    iminuit_fit = prepare_iminuit(nll, init_pars, data)
    iminuit_runtime, iminuit_bestfit = timeit(iminuit_fit)
    print("iminuit - runtime [s]:", iminuit_runtime)

    optimistix_migrad_fit = prepare_optimistix_migrad(nll, init_pars, data)
    optimistix_migrad_runtime, optimistix_migrad_bestfit = timeit(optimistix_migrad_fit)
    print("optimistix.migrad - runtime [s]:", optimistix_migrad_runtime)

    labels = model.config.par_names
    rel_diffs = (
        jnp.round(
            (iminuit_bestfit - optimistix_migrad_bestfit) / optimistix_migrad_bestfit, 2
        )
        * 100
    )
    print("Difference in bestfit parameters:")
    for name, iminuit_, opt_migrad_, diff in zip(
        labels,
        iminuit_bestfit.tolist(),
        optimistix_migrad_bestfit.tolist(),
        rel_diffs.tolist(),
    ):
        print(
            f"{name:>45}: iminuit={iminuit_:.4f} vs optimistix.migrad={opt_migrad_:.4f} (rel diff: {diff:.1f}%)"
        )

    # Output:
    # iminuit - runtime [s]: 1.2621246670605615
    # optimistix.migrad - runtime [s]: 0.366700749960728
    # Difference in bestfit parameters:
    #                             EG_RESOLUTION_ALL: iminuit=0.0005 vs optimistix.migrad=0.0005 (rel diff: 0.0%)
    #                                  EG_SCALE_ALL: iminuit=-0.0051 vs optimistix.migrad=-0.0052 (rel diff: -0.0%)
    #    EL_EFF_ChargeIDSel_TOTAL_1NPCOR_PLUS_UNCOR: iminuit=-0.0004 vs optimistix.migrad=-0.0004 (rel diff: -0.0%)
    #             EL_EFF_ID_TOTAL_1NPCOR_PLUS_UNCOR: iminuit=-0.0176 vs optimistix.migrad=-0.0176 (rel diff: -0.0%)
    #            EL_EFF_Iso_TOTAL_1NPCOR_PLUS_UNCOR: iminuit=-0.0234 vs optimistix.migrad=-0.0234 (rel diff: -0.0%)
    #           EL_EFF_Reco_TOTAL_1NPCOR_PLUS_UNCOR: iminuit=-0.0020 vs optimistix.migrad=-0.0020 (rel diff: -0.0%)
    #     EL_EFF_TriggerEff_TOTAL_1NPCOR_PLUS_UNCOR: iminuit=-0.0004 vs optimistix.migrad=-0.0004 (rel diff: -0.0%)
    #        EL_EFF_Trigger_TOTAL_1NPCOR_PLUS_UNCOR: iminuit=-0.0016 vs optimistix.migrad=-0.0016 (rel diff: -0.0%)
    #                          FT_EFF_B_systematics: iminuit=-0.0059 vs optimistix.migrad=-0.0059 (rel diff: 0.0%)
    #                          FT_EFF_C_systematics: iminuit=0.0281 vs optimistix.migrad=0.0281 (rel diff: -0.0%)
    #                      FT_EFF_Light_systematics: iminuit=-0.0402 vs optimistix.migrad=-0.0402 (rel diff: -0.0%)
    #                          FT_EFF_extrapolation: iminuit=-0.0054 vs optimistix.migrad=-0.0054 (rel diff: -0.0%)
    #               FT_EFF_extrapolation_from_charm: iminuit=0.0085 vs optimistix.migrad=0.0085 (rel diff: -0.0%)
    #      JET_EtaIntercalibration_NonClosure_highE: iminuit=-0.0007 vs optimistix.migrad=-0.0007 (rel diff: -0.0%)
    #     JET_EtaIntercalibration_NonClosure_negEta: iminuit=0.0174 vs optimistix.migrad=0.0174 (rel diff: -0.0%)
    #     JET_EtaIntercalibration_NonClosure_posEta: iminuit=-0.0005 vs optimistix.migrad=-0.0005 (rel diff: -0.0%)
    #                           JET_Flavor_Response: iminuit=0.0875 vs optimistix.migrad=0.0875 (rel diff: -0.0%)
    #                               JET_GroupedNP_1: iminuit=-0.1321 vs optimistix.migrad=-0.1321 (rel diff: -0.0%)
    #                               JET_GroupedNP_2: iminuit=-0.0923 vs optimistix.migrad=-0.0923 (rel diff: 0.0%)
    #                               JET_GroupedNP_3: iminuit=0.0240 vs optimistix.migrad=0.0240 (rel diff: -0.0%)
    #                              JET_JER_DataVsMC: iminuit=-0.0267 vs optimistix.migrad=-0.0268 (rel diff: -1.0%)
    #                         JET_JER_EffectiveNP_1: iminuit=-0.2798 vs optimistix.migrad=-0.2796 (rel diff: 0.0%)
    #                         JET_JER_EffectiveNP_2: iminuit=-0.2165 vs optimistix.migrad=-0.2157 (rel diff: 0.0%)
    #                         JET_JER_EffectiveNP_3: iminuit=0.1373 vs optimistix.migrad=0.1392 (rel diff: -1.0%)
    #                         JET_JER_EffectiveNP_4: iminuit=-0.0262 vs optimistix.migrad=-0.0262 (rel diff: -0.0%)
    #                         JET_JER_EffectiveNP_5: iminuit=-0.0102 vs optimistix.migrad=-0.0103 (rel diff: -0.0%)
    #                         JET_JER_EffectiveNP_6: iminuit=-0.0168 vs optimistix.migrad=-0.0168 (rel diff: -0.0%)
    #                 JET_JER_EffectiveNP_7restTerm: iminuit=-0.0387 vs optimistix.migrad=-0.0388 (rel diff: -0.0%)
    #                             JET_JvtEfficiency: iminuit=-0.0002 vs optimistix.migrad=-0.0000 (rel diff: 338.0%)
    #                          MET_SoftTrk_ResoPara: iminuit=0.0189 vs optimistix.migrad=0.0189 (rel diff: -0.0%)
    #                          MET_SoftTrk_ResoPerp: iminuit=-0.0396 vs optimistix.migrad=-0.0397 (rel diff: -0.0%)
    #                             MET_SoftTrk_Scale: iminuit=-0.0615 vs optimistix.migrad=-0.0616 (rel diff: -0.0%)
    #                         MUON_EFF_BADMUON_STAT: iminuit=-0.0004 vs optimistix.migrad=-0.0004 (rel diff: -0.0%)
    #                          MUON_EFF_BADMUON_SYS: iminuit=-0.0004 vs optimistix.migrad=-0.0004 (rel diff: -0.0%)
    #                             MUON_EFF_ISO_STAT: iminuit=-0.0009 vs optimistix.migrad=-0.0009 (rel diff: -0.0%)
    #                              MUON_EFF_ISO_SYS: iminuit=-0.0029 vs optimistix.migrad=-0.0029 (rel diff: 0.0%)
    #                            MUON_EFF_RECO_STAT: iminuit=-0.0012 vs optimistix.migrad=-0.0012 (rel diff: -0.0%)
    #                             MUON_EFF_RECO_SYS: iminuit=-0.0049 vs optimistix.migrad=-0.0049 (rel diff: -0.0%)
    #                            MUON_EFF_TTVA_STAT: iminuit=-0.0007 vs optimistix.migrad=-0.0007 (rel diff: -0.0%)
    #                             MUON_EFF_TTVA_SYS: iminuit=-0.0006 vs optimistix.migrad=-0.0006 (rel diff: -0.0%)
    #                  MUON_EFF_TrigStatUncertainty: iminuit=-0.0010 vs optimistix.migrad=-0.0010 (rel diff: -0.0%)
    #                  MUON_EFF_TrigSystUncertainty: iminuit=-0.0012 vs optimistix.migrad=-0.0012 (rel diff: -0.0%)
    #                                       MUON_ID: iminuit=-0.0144 vs optimistix.migrad=-0.0144 (rel diff: 0.0%)
    #                                       MUON_MS: iminuit=0.0118 vs optimistix.migrad=0.0119 (rel diff: -1.0%)
    #                          MUON_SAGITTA_RESBIAS: iminuit=0.0058 vs optimistix.migrad=0.0058 (rel diff: 0.0%)
    #                              MUON_SAGITTA_RHO: iminuit=-0.0004 vs optimistix.migrad=-0.0004 (rel diff: -0.0%)
    #                                    MUON_SCALE: iminuit=-0.0023 vs optimistix.migrad=-0.0023 (rel diff: 0.0%)
    #                                     ttbar_FSR: iminuit=0.0937 vs optimistix.migrad=0.0936 (rel diff: 0.0%)
    #                                     ttbar_Gen: iminuit=0.3278 vs optimistix.migrad=0.3278 (rel diff: -0.0%)
    #                                ttbar_ISR_Down: iminuit=-0.0066 vs optimistix.migrad=-0.0066 (rel diff: 0.0%)
    #                                  ttbar_ISR_Up: iminuit=0.2758 vs optimistix.migrad=0.2758 (rel diff: 0.0%)
    #                                      ttbar_PS: iminuit=0.3375 vs optimistix.migrad=0.3375 (rel diff: -0.0%)
    #                                   Z_theory_SR: iminuit=0.0530 vs optimistix.migrad=0.0531 (rel diff: -0.0%)
    #                                          lumi: iminuit=1.0000 vs optimistix.migrad=1.0000 (rel diff: -0.0%)
    #                                          mu_z: iminuit=1.0225 vs optimistix.migrad=1.0226 (rel diff: -0.0%)
    #                                        mu_SIG: iminuit=-6.5191 vs optimistix.migrad=-6.5166 (rel diff: 0.0%)
    #                                      mu_ttbar: iminuit=1.4727 vs optimistix.migrad=1.4725 (rel diff: 0.0%)
    #                                        SigRad: iminuit=0.0000 vs optimistix.migrad=0.0000 (rel diff: -91.0%)
    #                                    ttH_theory: iminuit=-0.0008 vs optimistix.migrad=-0.0008 (rel diff: -1.0%)
    #                                    ttZ_theory: iminuit=-0.0019 vs optimistix.migrad=-0.0019 (rel diff: -0.0%)
    #                        staterror_CRtt_cuts[0]: iminuit=0.9995 vs optimistix.migrad=0.9995 (rel diff: 0.0%)
    #                         staterror_CRz_cuts[0]: iminuit=0.9992 vs optimistix.migrad=0.9992 (rel diff: 0.0%)
    #                      staterror_SR_metsigST[0]: iminuit=1.0087 vs optimistix.migrad=1.0087 (rel diff: -0.0%)
    #                      staterror_SR_metsigST[1]: iminuit=0.9902 vs optimistix.migrad=0.9902 (rel diff: 0.0%)
    #                      staterror_SR_metsigST[2]: iminuit=1.0048 vs optimistix.migrad=1.0048 (rel diff: 0.0%)
    #                      staterror_SR_metsigST[3]: iminuit=0.9994 vs optimistix.migrad=0.9994 (rel diff: -0.0%)
