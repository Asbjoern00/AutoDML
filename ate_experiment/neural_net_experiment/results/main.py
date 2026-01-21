import main_DOPE_ries_netSepNets, main_DOPE_riesz_netOutcome, main_DOPE_riesz_netRieszInformed
if __name__ == '__main__':
    for n_shared in [2]:
        main_DOPE_riesz_netRieszInformed.run(n_shared)
#        main_DOPE_riesz_netOutcome.run(n_shared)

    main_DOPE_ries_netSepNets.run()
