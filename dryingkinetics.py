import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os, cv2, chilicv
from scipy.stats import linregress
"""Code for data analysis of chili pepper drying dataset."""

class DryingKinetics:
    def __init__(self):
        pass

    def load_data(self, data_path):
        dataset = pd.read_csv(data_path)
        return dataset
    
    def computeMR(self, ds, temp, init_moisture, path):
        # Create a new column with the row-by-row average
        ds['mean'] = ds[['rept_1', 'rept_2', 'rept_3']].mean(axis=1)
        min_val = ds[['rept_1', 'rept_2', 'rept_3']].min().min()
        # convert time from minutes to hours
        ds["time_hr"] = ds["time"] / 60
        
        # Plot drying time vs mass of dried chili pepper
        fig = plt.figure(figsize=(6,5))
        plt.scatter(ds["time"], ds["mean"])
        plt.plot(ds["time"], ds["mean"])
        plt.title("Plot of Drying Time vs Weight of Dried Chili Pepper")
        plt.xlabel("Drying time")
        plt.ylabel("Weight")
        # plt.show()
        plt.savefig(f"{path}_plot_weight.png")

        # Compute initial moisture content (mean)
        ds.loc[0, "Wm"] = ds.loc[0, "mean"] * init_moisture
        for i in range(1, len(ds-1)):
            ds.loc[i, "Wm"] = ds.loc[i-1, "Wm"] - (ds.loc[i-1, "mean"] - ds.loc[i, "mean"])
        ds["Ws"]= ds["mean"] - ds["Wm"] 
        Ws =ds["mean"] - ds["Wm"]
        ds['Ws'] = Ws
        ds["m_wb"] = ds["Wm"]/(ds.loc[0,"Wm"] + ds["Ws"])
        ds["M_db"] = ds["Wm"]/ds["Ws"]

        # Compute initial moisture content (rept_1)
        ds.loc[0, "Wm1"] = ds.loc[0, "rept_1"] * init_moisture
        for i in range(1, len(ds-1)):
            ds.loc[i, "Wm1"] = ds.loc[i-1, "Wm1"] - (ds.loc[i-1, "rept_1"] - ds.loc[i, "rept_1"])
        ds["Ws1"]= ds["rept_1"] - ds["Wm1"] 
        Ws1 =ds["rept_1"] - ds["Wm1"]
        ds['Ws1'] = Ws1
        ds["m_wb1"] = ds["Wm1"]/(ds.loc[0,"Wm1"] + ds["Ws1"])
        ds["M_db1"] = ds["Wm1"]/ds["Ws1"]

        # Compute initial moisture content (rept_2)
        ds.loc[0, "Wm2"] = ds.loc[0, "rept_2"] * init_moisture
        for i in range(1, len(ds-1)):
            ds.loc[i, "Wm2"] = ds.loc[i-1, "Wm2"] - (ds.loc[i-1, "rept_2"] - ds.loc[i, "rept_2"])
        ds["Ws2"]= ds["rept_2"] - ds["Wm2"] 
        Ws2 =ds["rept_2"] - ds["Wm2"]
        ds['Ws2'] = Ws2
        ds["m_wb2"] = ds["Wm2"]/(ds.loc[0,"Wm2"] + ds["Ws2"])
        ds["M_db2"] = ds["Wm2"]/ds["Ws2"]

        # Compute initial moisture content (rept_3)
        ds.loc[0, "Wm3"] = ds.loc[0, "rept_3"] * init_moisture
        for i in range(1, len(ds-1)):
            ds.loc[i, "Wm3"] = ds.loc[i-1, "Wm3"] - (ds.loc[i-1, "rept_3"] - ds.loc[i, "rept_3"])
        ds["Ws3"]= ds["rept_3"] - ds["Wm3"] 
        Ws3 =ds["rept_3"] - ds["Wm3"]
        ds['Ws3'] = Ws3
        ds["m_wb3"] = ds["Wm3"]/(ds.loc[0,"Wm3"] + ds["Ws3"])
        ds["M_db3"] = ds["Wm3"]/ds["Ws3"]

        # Plot moisture dry basis
        fig = plt.figure(figsize=(6,5))
        plt.scatter(ds["time"], ds["M_db"])
        plt.plot(ds["time"], ds["M_db"])
        plt.title("Plot of Drying Time vs Moisture Content (dry basis)")
        plt.xlabel("Drying time")
        plt.ylabel("Moisture Content (dry basis)")
        plt.savefig(f"{path}_plot_M_db.png")

        # Compute Drying Rate, time in hours, mean
        ds.loc[0,"dM/dt"] = 0
        for i in range(1, len(ds-1)):
            ds.loc[i,"dM/dt"] = (ds.loc[i-1, "M_db"]-ds.loc[i, "M_db"])/(ds.loc[i, "time_hr"]-ds.loc[i-1, "time_hr"])
        # Plot Drying Rate vs Moisture Content
        fig = plt.figure(figsize=(6,5))
        plt.scatter(ds["M_db"][:-1], ds["dM/dt"][:-1])
        plt.plot(ds["M_db"][:-1], ds["dM/dt"][:-1])
        plt.title("Plot of Drying Rate")
        plt.xlabel("M_db")
        plt.ylabel("dM/dt")
        plt.savefig(f"{path}_plot_DR.png")

        # Drying Rate (Rc)
        idx_max = ds["dM/dt"].idxmax()    
        # Critical Moisture Content (Xc)
        M_db_max = ds.loc[idx_max, "M_db"]

        # Moisture Ratio
        ds["MR_obs"] = ds["M_db"] / ds.loc[0,"M_db"]
        ds["MR_obs1"] = ds["M_db1"] / ds.loc[0,"M_db1"]
        ds["MR_obs2"] = ds["M_db2"] / ds.loc[0,"M_db2"]
        ds["MR_obs3"] = ds["M_db3"] / ds.loc[0,"M_db3"]
    
        # Save the updated dataset to a new CSV file
        ds.to_csv(f"{path}_updated.csv", index=True)
    
        return ds
    
    def moisture(self, ds, temp, path):

        # The models derived from Newton's Law of Cooling
        def lewis(t, k):                         return np.exp(-k*t)
        def page(t, k, n):                              return np.exp(-k*t**n)
        def modified_page(t, k, n):                     return np.exp(-(k*t)**n)
        # The models derived from Fick's Second Law of Diffusion
        def henderson_pabis(t, a, k):                   return a*np.exp(-k*t)
        def logarithmic(t, a, k, c):                    return a*np.exp(-k*t) + c
        def midilli(t, a, k, n, b):                     return a*np.exp(-k*t**n) + b*t
        def modified_midilli(t, a, k, n, b):            return np.exp(-k*t**n) + b*t
        def modified_midilli_kucuk(t, k, a, n, b):      return a*np.exp(-k*t**n) + b
        def demir(t, k, a, b, n):                       return a*np.exp(-k*t)**n + b
        def verma(t, a, k, g):                          return a*np.exp(-k*t) + (1-a)*np.exp(-g*t)
        def two_term(t, a, k0, b, k1):                  return a*np.exp(-k0*t) + b*np.exp(-k1*t)
        def two_term_exponential(t, k, a):              return a*np.exp(-k*t) + (1-a)*np.exp(-k*a*t)
        def diffusion(t, k, a, b):                      return a*np.exp(-k*t) + (1-a)*np.exp(-k*b*t)
        def modified_henderson_pabis(t, a, k, g, c, h): return a*np.exp(-k*t) + (1-a)*np.exp(-g*t) + c*np.exp(-h*t)
        def jena_das(t, a, k, b, c):                    return a*np.exp(-k*t + b*np.sqrt(t)) + c
        def logistic(t, a, k, b):                       return a/(1 + b * np.exp(k*t))
        # Empirical models
        # def thompson(MR, a, b):                         return a*np.log(MR) + b*(np.log(MR)**2)
        def wang_singh(t, a, b):                        return 1 + a*t + b*t**2
        def parabolic(t, a, b, c):                      return a - b*t - c*t**2
        def weibull(t, a, b):                           return np.exp(- (t/b)**a)
        def aghbashlo(t, k1, k2):                       return np.exp((-k1*t)/(1 + k2*t))
        def kaleemullah(t, temp, a, b, c, d):           return np.exp(-(a*temp + b)*t**(c*temp+d))
        def kaleemullah_fixedT(t, a, b, c, d):          return kaleemullah(t, temp, a, b, c, d)
        
        # The models derived from Newton's Law of Cooling
        ## Lewis
        params_l, _ = curve_fit(lewis, ds["time"], ds["MR_obs"], p0=(0.001))
        k_l = params_l; print("Lewis \nk =", k_l)
        ## Page
        params_pg, _ = curve_fit(page, ds["time"], ds["MR_obs"], p0=(0.001,1))
        k_pg, n_pg = params_pg; print("Page \nk =", k_pg, "n =", n_pg)
        ## Modified Page
        params_mp, _ = curve_fit(modified_page, ds["time"], ds["MR_obs"], p0=(0.001,1),
                         bounds=([0,0],[np.inf,5]))
        k_mp, n_mp = params_mp; print("Modified Page \nk =", k_mp, "n =", n_mp)
        
        # The models derived from Fick's Second Law of Diffusion
        ## Henderson-Pabis
        params_hp, _ = curve_fit(henderson_pabis, ds["time"], ds["MR_obs"], p0=(1,0.001))
        a_hp, k_hp = params_hp; print("Henderson-Pabis \na =", a_hp, "k =", k_hp)
        ## Logarithmic
        params_log, _ = curve_fit(logarithmic, ds["time"], ds["MR_obs"], p0=(1,0.001,0))
        a_log, k_log, c_log = params_log; print("Logarithmic \na =", a_log, "k =", k_log, "c =", c_log)
        ## Midilli
        params_md, _ = curve_fit(midilli, ds["time"], ds["MR_obs"], p0=(1,0.001,1,1e-5),
                         bounds=([0,0,0,0],[np.inf,np.inf,5,1]))
        a_md, k_md, n_md, b_md = params_md; print("Midilli \na =", a_md, "k =", k_md, "n =", n_md, "b =", b_md)
        ## Modified Midilli
        params_mm, _ = curve_fit(modified_midilli, ds["time"], ds["MR_obs"], p0=(1,0.001,1,1e-5),
                         bounds=([0,0,0,0],[np.inf,np.inf,5,1]))
        a_mm, k_mm, n_mm, b_mm = params_mm; print("Modified Midilli \na =", a_mm, "k =", k_mm, "n =", n_mm, "b =", b_mm)
        ## Modified Midilli-Kucuk
        params_mmk, _ = curve_fit(modified_midilli_kucuk, ds["time"], ds["MR_obs"], p0=(1,0.001,1,1e-5),
                         bounds=([0,0,0,0],[np.inf,np.inf,5,1]))
        k_mmk, a_mmk, b_mmk, n_mmk = params_mmk; print("Modified Midilli-Kucuk \nk =", k_mmk, "a =", a_mmk, "b =", b_mmk, "n =", n_mmk)
        ## Demir
        params_dm, _ = curve_fit(demir, ds["time"], ds["MR_obs"], p0=(0.001,1,1,1),
                         bounds=([0,0,0,0],[np.inf,np.inf,5,1]))
        k_dm, a_dm, b_dm, n_dm = params_dm; print("Demir \nk =", k_dm, "a =", a_dm, "b =", b_dm, "n =", n_dm)    
        ## Verma
        params_vm, _ = curve_fit(verma, ds["time"], ds["MR_obs"], p0=(0.5,0.001,0.001),
                         bounds=([0,0,0],[1,np.inf,np.inf]))
        a_vm, k_vm, g_vm = params_vm; print("Verma \na =", a_vm, "k =", k_vm, "g =", g_vm)
        ## Two-term
        params_tt, _ = curve_fit(two_term, ds["time"], ds["MR_obs"], p0=(0.5,0.001,0.5,0.0001),
                         bounds=([0,0,0,0],[1,np.inf,1,np.inf]))
        a_tt, k0_tt, b_tt, k1_tt = params_tt; print("Two-term \na =", a_tt, "k0 =", k0_tt, "b =", b_tt, "k1 =", k1_tt)
        ## Two-term exponential
        params_ttx, _ = curve_fit(two_term_exponential, ds["time"], ds["MR_obs"], p0=(0.001,0.7), bounds=([0,0],[np.inf,1]), maxfev=20000)
        k_ttx, a_ttx = params_ttx; print("Two-term exponential \nk =", k_ttx, "a =", a_ttx)
        ## Diffusion
        params_diffusion, _ = curve_fit(diffusion, ds["time"], ds["MR_obs"], p0=(0.001, 0.5, 0.5), bounds=([0,0,0],[np.inf,1,1]), maxfev=5000)
        k_diffusion, a_diffusion, b_diffusion = params_diffusion; print("Diffusion \nk =", k_diffusion, "a =", a_diffusion, "b =", b_diffusion)
        ## Jena-Das
        params_jd, _ = curve_fit(jena_das, ds["time"], ds["MR_obs"], p0=(1,0.001,1e-5,1e-5))
        a_jd, k_jd, b_jd, c_jd = params_jd; print("Jena-Das \na =", a_jd, "k =", k_jd, "b =", b_jd, "c =", c_jd)
        ## Modified Henderson-Pabis
        params_mhp, _ = curve_fit(modified_henderson_pabis, ds["time"], ds["MR_obs"], p0=(0.8, 0.01, 0.005, 0.05, 0.001),
                          bounds=([0,0,0,0,0],[1,np.inf,np.inf,1,np.inf]), maxfev = 10000)
        a_mhp, k_mhp, g_mhp, c_mhp, h_mhp = params_mhp; print("Modified Henderson-Pabis \na =", a_mhp, "k =", k_mhp, "g =", g_mhp, "c =", c_mhp, "h =", h_mhp)
        ## Logistic
        params_lg, _ = curve_fit(logistic, ds["time"], ds["MR_obs"], p0=(1,0.001,1e-5),
                         bounds=([0,0,0],[np.inf,np.inf,np.inf]))
        a_lg, k_lg, b_lg = params_lg; print("Logistic \na =", a_lg, "k =", k_lg, "b =", b_lg)

        # Empirical models
        ## Thompson
        # params_th, _ = curve_fit(thompson, ds["MR_obs"], ds["time"], p0=(-100,-100))
        # a, b = params_th; print("Thompson parameters: a =", a, "b =", b)
        ## Weibull
        params_wb, _ = curve_fit(weibull, ds["time"], ds["MR_obs"], p0=(1,100),
                         bounds=([0,0],[np.inf,np.inf]))
        a_wb, b_wb = params_wb; print("Weibull\na =", a_wb, "b =", b_wb)
        ## Wang & Singh
        params_ws, _ = curve_fit(wang_singh, ds["time"], ds["MR_obs"], p0=(-0.001,-1e-6))
        a_ws, b_ws = params_ws; print("Wang & Singh \na =", a_ws, "b =", b_ws)
        ## Parabolic
        params_par, _ = curve_fit(parabolic, ds["time"], ds["MR_obs"], p0=(1,0.001,1e-5),
                         bounds=([0,0,0],[np.inf,np.inf,np.inf]))
        a_par, b_par, c_par = params_par; print("Parabolic \na =", a_par, "b =", b_par, "c =", c_par)
        ## Aghbashlo
        params_ag, _ = curve_fit(aghbashlo, ds["time"], ds["MR_obs"], p0=(0.001,0.0001),
                         bounds=([0,0],[np.inf,np.inf]))
        k1_ag, k2_ag = params_ag; print("Aghbashlo \nk1 =", k1_ag, "k2 =", k2_ag)
        ## Kaleemullah
        params_kal, _ = curve_fit(kaleemullah_fixedT, ds["time"], ds["MR_obs"], p0=(0.001,0.0001,0.001,0.001),
                         bounds=([0,0,0,0],[np.inf,np.inf,np.inf,np.inf]), maxfev = 10000)
        a_kal, b_kal, c_kal, d_kal = params_kal; print("Kaleemullah \na =", a_kal, "b =", b_kal, "c =", c_kal, "d =", d_kal)
        
        # Save results on txt
        results = []
        results.append(f"Lewis: k={k_l}")
        results.append(f"Page: k={k_pg}, n={n_pg}")
        results.append(f"Modified Page: k={k_mp}, n={n_mp}")
        results.append(f"Henderson-Pabis: a={a_hp}, k={k_hp}")
        results.append(f"Logarithmic: a={a_log}, k={k_log}, c={c_log}")
        results.append(f"Midilli: a={a_md}, k={k_md}, n={n_md}, b={b_md}")
        results.append(f"Modified Midilli: a={a_mm}, k={k_mm}, n={n_mm}, b={b_mm}")
        results.append(f"Modified Midilli-Kucuk: k={k_mmk}, a={a_mmk}, b={b_mmk}, n={n_mmk}")
        results.append(f"Demir: k={k_dm}, a={a_dm}, b={b_dm}, n={n_dm}")
        results.append(f"Verma: a={a_vm}, k={k_vm}, g={g_vm}")
        results.append(f"Two-term: a={a_tt}, k0={k0_tt}, b={b_tt}, k1={k1_tt}")
        results.append(f"Two-term exponential: k={k_ttx}, a={a_ttx}")
        results.append(f"Diffusion: k={k_diffusion}, a={a_diffusion}, b={b_diffusion}")
        results.append(f"Jena-Das: a={a_jd}, k={k_jd}, b={b_jd}, c={c_jd}")
        results.append(f"Modified Henderson-Pabis: a={a_mhp}, k={k_mhp}, g={g_mhp}, c={c_mhp}, h={h_mhp}")
        results.append(f"Logistic: a={a_lg}, k={k_lg}, b={b_lg}")
        results.append(f"Weibull: a={a_wb}, b={b_wb}")
        results.append(f"Wang & Singh: a={a_ws}, b={b_ws}")
        results.append(f"Parabolic: a={a_par}, b={b_par}, c={c_par}")
        results.append(f"Aghbashlo: k1={k1_ag}, k2={k2_ag}")
        results.append(f"Kaleemullah: a={a_kal}, b={b_kal}, c={c_kal}, d={d_kal}")
        
        # Simpan ke file teks
        with open(f"{path}models_constants.txt", "w") as f:
            for line in results:
                f.write(line + "\n")

        # Predictions
        # Derived from Newton's Law of Cooling
        ds["MR_nl"] = lewis(ds["time"], *params_l)
        ds["MR_pg"] = page(ds["time"], *params_pg)
        ds["MR_mp"]= modified_page(ds["time"], *params_mp)
        # Derived from Fick's Second Law of Diffusion
        ds["MR_hp"] = henderson_pabis(ds["time"], *params_hp)
        ds["MR_log"] = logarithmic(ds["time"], *params_log)
        ds["MR_md"] = midilli(ds["time"], *params_md)
        ds["MR_mm"] = modified_midilli(ds["time"], *params_mm)
        ds["MR_mmk"] = modified_midilli_kucuk(ds["time"], *params_mmk)
        ds["MR_dm"] = demir(ds["time"], *params_dm)
        ds["MR_vm"] = verma(ds["time"], *params_vm)
        ds["MR_tt"] = two_term(ds["time"], *params_tt)
        ds["MR_ttx"] = two_term_exponential(ds["time"], *params_ttx)
        ds["MR_diffusion"] = diffusion(ds["time"], *params_diffusion)
        ds["MR_jd"] = jena_das(ds["time"], *params_jd)
        ds["MR_mhp"] = modified_henderson_pabis(ds["time"], *params_mhp)
        ds["MR_lg"] = logistic(ds["time"], *params_lg)
        # Empirical 
        ds["MR_ws"] = wang_singh(ds["time"], *params_ws)
        # ds["time_th"] = thompson(ds["MR_obs"], *params_th)
        ds["MR_par"] = parabolic(ds["time"], *params_par)
        ds["MR_wb"] = weibull(ds["time"], *params_wb)
        ds["MR_ag"] = aghbashlo(ds["time"], *params_ag)
        ds["MR_kal"] = kaleemullah(ds["time"], temp, *params_kal)

        # Goodness-of-fit function
        def metrics(obs, pred):
            SS_res = np.sum((obs - pred)**2)
            SS_tot = np.sum((obs - np.mean(obs))**2)
            R2 = 1 - SS_res/SS_tot
            RMSE = np.sqrt(SS_res/len(obs))
            chi2 = np.sum(((obs - pred)**2)/(pred+1e-8))
            return R2, RMSE, chi2
        
        # Collect results
        results = {
            "Lewis": metrics(ds["MR_obs"], ds["MR_nl"]),
            "Page": metrics(ds["MR_obs"], ds["MR_pg"]),
            "Modified Page": metrics(ds["MR_obs"], ds["MR_mp"]),

            "Henderson-Pabis": metrics(ds["MR_obs"], ds["MR_hp"]),
            "Logarithmic": metrics(ds["MR_obs"], ds["MR_log"]),
            "Midilli": metrics(ds["MR_obs"], ds["MR_md"]),
            "Modified Midilli": metrics(ds["MR_obs"], ds["MR_mm"]),
            "Modified Midilli-Kucuk": metrics(ds["MR_obs"], ds["MR_mmk"]),
            "Demir": metrics(ds["MR_obs"], ds["MR_dm"]),
            "Verma": metrics(ds["MR_obs"], ds["MR_vm"]),
            "Two-term": metrics(ds["MR_obs"], ds["MR_tt"]),
            "Two-term Exponential": metrics(ds["MR_obs"], ds["MR_ttx"]),
            "Diffusion": metrics(ds["MR_obs"], ds["MR_diffusion"]),
            "Jena-Das": metrics(ds["MR_obs"], ds["MR_jd"]),
            "Modified Henderson-Pabis": metrics(ds["MR_obs"], ds["MR_mhp"]),
            "Logistic": metrics(ds["MR_obs"], ds["MR_lg"]),

            "Wang-Singh": metrics(ds["MR_obs"], ds["MR_ws"]),
            "Parabolic": metrics(ds["MR_obs"], ds["MR_par"]),
            "Weibull": metrics(ds["MR_obs"], ds["MR_wb"]),
            "Aghbashlo": metrics(ds["MR_obs"], ds["MR_ag"]),
            #"Thompson": metrics(ds["time"], ds["time_th"]),
            "Kaleemullah": metrics(ds["MR_obs"], ds["MR_kal"]),
        }

        # Print table
        print(f"{'Model':<20}{'R²':<15}{'RMSE':<15}{'Chi²':<15}")
        for model, (R2, RMSE, chi2) in results.items():
            print(f"{model:<20}{R2:<15.4f}{RMSE:<15.4f}{chi2:<15.4f}")

        # Convert to DataFrame
        df_results = pd.DataFrame(
            [(model, R2, RMSE, chi2) for model, (R2, RMSE, chi2) in results.items()],
            columns=["Model", "R2", "RMSE", "Chi2"])
        # Save to CSV
        df_results.to_csv(f"{path}_statResults.csv", index=False)
        
        # Plot observed data
        plt.figure(figsize=(6,5))
        plt.scatter(ds["time"], ds["MR_obs"], color="black", label="Observed MC (db)", s=25)
        # Plot predictions for each model
        # plt.plot(ds["time"], ds["MR_nl"], label="Newton-Lewis", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_pg"], label="Page", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_mp"], label="Modified Page", lw = 2.0)

        # plt.plot(ds["time"], ds["MR_hp"], label="Henderson-Pabis", lw = 2.0)
        plt.plot(ds["time"], ds["MR_log"], label="Logarithmic", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_md"], label="Midilli", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_mm"], label="Modified Midilli", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_mmk"], label="Modified Midilli-Kucuk", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_dm"], label="Demir", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_vm"], label="Verma", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_tt"], label="Two-term", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_ttx"], label="Two-term Exponential", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_diffusion"], label="Diffusion", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_mhp"], label="Mod Hend.-Pabis", lw = 2.0)
        plt.plot(ds["time"], ds["MR_jd"], label="Jena-Das", lw = 2.0)
        plt.plot(ds["time"], ds["MR_lg"], label="Logistic", lw = 2.0)

        # plt.plot(ds["time"], ds["MR_par"], label="Parabolic", lw = 2.0)
        plt.plot(ds["time"], ds["MR_ws"], label="Wang-Singh", lw = 2.0)        
        # plt.plot(ds["time"], ds["MR_wb"], label="Weibull", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_ag"], label="Aghbashlo", lw = 2.0)
        # plt.plot(ds["time"], ds["MR_kal"], label="Kaleemullah", lw = 2.0)   

        plt.xlabel("Time (min)", fontsize=11)
        plt.ylabel("Moisture Ratio (MR)", fontsize=11)
        plt.title("Observed vs Predicted MR for Drying Models")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{path}_plot_MRpred.png")

        # Save the updated dataset to a new CSV file
        ds.to_csv(f"{path}_updated.csv", index=True)
        return ds

    def plot_MRxDR(self, data1, data2, data3, out1, out2, out3):
        # Plot moisture content versus time for different temperatures
        plt.figure(figsize=(6, 5))
    
        # Plot data for each temperature with error bar
        if data1 is not None:
            plt.errorbar(data1['time'], data1['MR_obs'], fmt='o-', label='60°C', color='black', markerfacecolor='white', markersize=6)
        if data2 is not None:
            plt.errorbar(data2['time'], data2['MR_obs'], fmt='s-', label='70°C', color='black', markerfacecolor='white', markersize=6)
        if data3 is not None:
            plt.errorbar(data3['time'], data3['MR_obs'], fmt='^-', label='80°C', color='black', markerfacecolor='white', markersize=6)
        # add line at y = 0.05
        plt.axhline(y=0.00, color='black', linestyle='-')
        # Customize the plot
        plt.xlabel('Time (min)', fontsize=12)
        plt.ylabel('Moisture Ratio', fontsize=12)
        # plt.title('Moisture Ratio vs Time at Different Temperatures', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{out1}", dpi=300, bbox_inches='tight')

        # PLOT DRYING RATE vs MOISTURE RATIO
        plt.figure(figsize=(6,5))
        # Plot data for each temperature
        if data1 is not None:
            plt.plot(data1['M_db'], data1['dM/dt'], 'o-', label='60°C', color='black', markerfacecolor='white', markersize=6)
        if data2 is not None:
            plt.plot(data2['M_db'], data2['dM/dt'], 's-', label='70°C', color='black', markerfacecolor='white', markersize=6)
        if data3 is not None:
            plt.plot(data3['M_db'], data3['dM/dt'], '^-', label='80°C', color='black', markerfacecolor='white', markersize=6)
        # Customize the plot
        plt.xlabel('Moisture Content (g water/g dry, d.b.)', fontsize=12)
        plt.ylabel('Drying Rate (g water/g dry/hour)', fontsize=12)
        # plt.title('Moisture Ratio vs Time at Different Temperatures', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{out2}", dpi=300, bbox_inches='tight')

        # PLOT DRYING RATE vs TIME (min)
        plt.figure(figsize=(6, 5))
        # Plot data for each temperature
        if data1 is not None:
            plt.plot(data1['time'], data1['dM/dt'], 'o-', label='60°C', color='black', markerfacecolor='white', markersize=6)
        if data2 is not None:
            plt.plot(data2['time'], data2['dM/dt'], 's-', label='70°C', color='black', markerfacecolor='white', markersize=6)
        if data3 is not None:
            plt.plot(data3['time'], data3['dM/dt'], '^-', label='80°C', color='black', markerfacecolor='white', markersize=6)
        # Customize the plot
        plt.xlabel('Time (min)', fontsize=12)
        plt.ylabel('Drying Rate (g water/g dry/hour)', fontsize=12)
        # plt.title('Moisture Ratio vs Time at Different Temperatures', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{out3}", dpi=300, bbox_inches='tight')

    def MCDryAttributes(self, data1, data2, data3,output):
        # Call MR data (mean)
        MR60 = data1["MR_obs"]; MR70 = data2["MR_obs"]; MR80 = data3["MR_obs"]
        # Call MR data (rept_1)
        MR60_1 = data1["MR_obs1"]; MR70_1 = data2["MR_obs1"]; MR80_1 = data3["MR_obs1"]
        # Call MR data (rept_2)
        MR60_2 = data1["MR_obs2"]; MR70_2 = data2["MR_obs2"]; MR80_2 = data3["MR_obs2"]
        # Call MR data (rept_3)
        MR60_3 = data1["MR_obs3"]; MR70_3 = data2["MR_obs3"]; MR80_3 = data3["MR_obs3"]
        # Call time data
        time60 = data1["time"]; time70 = data2["time"]; time80 = data3["time"]
        
        """Effective moisture diffusivity (Deff)"""

        # Filter MR > 0 (avoid log NaN), mean
        lnMR60 = np.log(MR60[MR60 > 0]); t60 = time60[MR60 > 0]
        lnMR70 = np.log(MR70[MR70 > 0]); t70 = time70[MR70 > 0]
        lnMR80 = np.log(MR80[MR80 > 0]); t80 = time80[MR80 > 0]
        # Filter MR > 0 (avoid log NaN), rept_1
        lnMR60_1 = np.log(MR60_1[MR60_1 > 0]); t60_1 = time60[MR60_1 > 0]
        lnMR70_1 = np.log(MR70_1[MR70_1 > 0]); t70_1 = time70[MR70_1 > 0]
        lnMR80_1 = np.log(MR80_1[MR80_1 > 0]); t80_1 = time80[MR80_1 > 0]
        # Filter MR > 0 (avoid log NaN), rept_2
        lnMR60_2 = np.log(MR60_2[MR60_2 > 0]); t60_2 = time60[MR60_2 > 0]
        lnMR70_2 = np.log(MR70_2[MR70_2 > 0]); t70_2 = time70[MR70_2 > 0]
        lnMR80_2 = np.log(MR80_2[MR80_2 > 0]); t80_2 = time80[MR80_2 > 0]
        # Filter MR > 0 (avoid log NaN), rept_3
        lnMR60_3 = np.log(MR60_3[MR60_3 > 0]); t60_3 = time60[MR60_3 > 0]
        lnMR70_3 = np.log(MR70_3[MR70_3 > 0]); t70_3 = time70[MR70_3 > 0]
        lnMR80_3 = np.log(MR80_3[MR80_3 > 0]); t80_3 = time80[MR80_3 > 0]
        
        # Find slope (k) from equation: ln(MR) = ln(8/π^2) - k t
        # Linear regression ln(MR) vs time, mean
        slope60, _, _, _, _ = linregress(t60, lnMR60)
        slope70, _, _, _, _ = linregress(t70, lnMR70)
        slope80, _, _, _, _ = linregress(t80, lnMR80)
        # Linear regression ln(MR) vs time, rept_1
        slope60_1, _, _, _, _ = linregress(t60_1, lnMR60_1)
        slope70_1, _, _, _, _ = linregress(t70_1, lnMR70_1)
        slope80_1, _, _, _, _ = linregress(t80_1, lnMR80_1)
        # Linear regression ln(MR) vs time, rept_2
        slope60_2, _, _, _, _ = linregress(t60_2, lnMR60_2)
        slope70_2, _, _, _, _ = linregress(t70_2, lnMR70_2)
        slope80_2, _, _, _, _ = linregress(t80_2, lnMR80_2)
        # Linear regression ln(MR) vs time, rept_3
        slope60_3, _, _, _, _ = linregress(t60_3, lnMR60_3)
        slope70_3, _, _, _, _ = linregress(t70_3, lnMR70_3)
        slope80_3, _, _, _, _ = linregress(t80_3, lnMR80_3)

        k60, k70, k80 = -slope60, -slope70, -slope80
        k60_1, k70_1, k80_1 = -slope60_1, -slope70_1, -slope80_1
        k60_2, k70_2, k80_2 = -slope60_2, -slope70_2, -slope80_2
        k60_3, k70_3, k80_3 = -slope60_3, -slope70_3, -slope80_3
        
        # Calculate effective diffusion coefficients (Deff), mean
        H = 1.19
        Deff60 = (k60 * (H)**2) / (np.pi**2)
        Deff70 = (k70 * (H)**2) / (np.pi**2)
        Deff80 = (k80 * (H)**2) / (np.pi**2)
        # Calculate effective diffusion coefficients (Deff), rept_1
        Deff60_1 = (k60_1 * (H)**2) / (np.pi**2)
        Deff70_1 = (k70_1 * (H)**2) / (np.pi**2)
        Deff80_1 = (k80_1 * (H)**2) / (np.pi**2)
        # Calculate effective diffusion coefficients (Deff), rept_2
        Deff60_2 = (k60_2 * (H)**2) / (np.pi**2)
        Deff70_2 = (k70_2 * (H)**2) / (np.pi**2)
        Deff80_2 = (k80_2 * (H)**2) / (np.pi**2)
        # Calculate effective diffusion coefficients (Deff), rept_3
        Deff60_3 = (k60_3 * (H)**2) / (np.pi**2)
        Deff70_3 = (k70_3 * (H)**2) / (np.pi**2)
        Deff80_3 = (k80_3 * (H)**2) / (np.pi**2)
        
        # Mean and standard deviation of Deff at each temperature 
        Deff_60 = np.array([Deff60_1, Deff60_2, Deff60_3])
        Deff_70 = np.array([Deff70_1, Deff70_2, Deff70_3])
        Deff_80 = np.array([Deff80_1, Deff80_2, Deff80_3])
        Deff_mean = np.array([np.mean(Deff_60), np.mean(Deff_70), np.mean(Deff_80)])
        Deff_sd = np.array([np.std(Deff_60), np.std(Deff_70), np.std(Deff_80)])
        print(f"Effective moisture diffusivity (Deff) at 60°C: {Deff_mean[0]} +/- {Deff_sd[0]}")
        print(f"Effective moisture diffusivity (Deff) at 70°C: {Deff_mean[1]} +/- {Deff_sd[1]}")
        print(f"Effective moisture diffusivity (Deff) at 80°C: {Deff_mean[2]} +/- {Deff_sd[2]}")
        
        """Activation Energy (Ea) using Arrhenius equation: Deff = Do * exp(-Ea / RT)"""
        R = 8.314 # J mol−1 K−1
        T = np.array([60, 70, 80]) + 273.15
    
        # Calculate Ea (mean)
        slope, intercept, _, _, _ = linregress(1/T, np.log(Deff_mean))
        Do = np.exp(intercept); Ea = -slope * R; Ea_kj = Ea/1000
        print(f"Pre-exponential factor (Do): {Do} m²/s")
        print(f"Activation Energy (Ea): {Ea_kj} kJ/mol")
        print(f"Equation: ln(Deff) = ln({Do}) - ({Ea}/{R})(1/T)")

        # plot ln(Deff) vs 1/T
        plt.figure(figsize=(6, 3))
        # experimental data
        plt.scatter(1/T, np.log(Deff_mean), color='black', label='Experimental data', marker='o', facecolors='white', s=60)
        # linear trendline
        x1 = np.linspace(1/T.max(), 1/T.min(), 100)
        y1 = slope * x1 + intercept
        plt.plot(x1, y1, color='black',  lw = 1.5, linestyle='--')
        # set intervals of x axis
        plt.xticks(np.linspace(1/T.min(), 1/T.max(), 5))
        plt.xlabel('1/T (K⁻¹)', fontsize=12)
        plt.ylabel('ln(Deff)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{output}", dpi=300, bbox_inches='tight')
        # R-squared
        ss_res = np.sum((np.log(Deff_mean) - (slope * (1/T) + intercept))**2)
        ss_tot = np.sum((np.log(Deff_mean) - np.mean(np.log(Deff_mean)))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R-squared: {r_squared}")

        # Change in Entalphy, ∆H=E_a-R (T+273.15)
        Delta_H60 = (Ea_kj - (R * (T[0] + 273.15))/1000)
        Delta_H70 = (Ea_kj - (R * (T[1] + 273.15))/1000)
        Delta_H80 = (Ea_kj - (R * (T[2] + 273.15))/1000)
        # Change in Entropy, ∆S=R (ln(D_0 )-ln(k_B/h_P )-ln(T+273.15)), in kJ/mol/T
        k_B = 1.381e-23 # Boltzmann constant (J/K)
        h_P = 6.626e-34 # Planck constant (J·s)
        kb_hp = k_B/h_P
        Delta_S60 = (R * (np.log(Do) - np.log(kb_hp) - np.log(T[0] + 273.15)))/1000
        Delta_S70 = (R * (np.log(Do) - np.log(kb_hp) - np.log(T[1] + 273.15)))/1000
        Delta_S80 = (R * (np.log(Do) - np.log(kb_hp) - np.log(T[2] + 273.15)))/1000
        # Change in Gibbs Free Energy, ∆G=∆H-T∆S, in kJ/mol
        Delta_G60 = Delta_H60 - (T[0] + 273.15) * Delta_S60
        Delta_G70 = Delta_H70 - (T[1] + 273.15) * Delta_S70
        Delta_G80 = Delta_H80 - (T[2] + 273.15) * Delta_S80
        
        print(f"Change in Entalphy (∆H) 60°C: {Delta_H60} kJ/mol")
        print(f"Change in Entropy (∆S) 60°C: {Delta_S60} kJ/mol/K")
        print(f"Change in Gibbs Free Energy (∆G) 60°C: {Delta_G60} kJ/mol")
        print(f"Change in Entalphy (∆H) 70°C: {Delta_H70} kJ/mol")
        print(f"Change in Entropy (∆S) 70°C: {Delta_S70} kJ/mol/K")
        print(f"Change in Gibbs Free Energy (∆G) 70°C: {Delta_G70} kJ/mol")
        print(f"Change in Entalphy (∆H) 80°C: {Delta_H80} kJ/mol")
        print(f"Change in Entropy (∆S) 80°C: {Delta_S80} kJ/mol/K")
        print(f"Change in Gibbs Free Energy (∆G) 80°C: {Delta_G80} kJ/mol")

    def average_Lab(self, input_folder, output_file):
        # List all CSV files in the folder
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

        if not csv_files:
            print("No CSV files found in the specified folder.")
            return

        # Initialize an empty DataFrame to hold the combined data
        all_data = []
        for file in csv_files:
            time_str = file[-7:-4]
            try:
                time_int = int(time_str)
            except ValueError:
                print(f"Skipping file '{file}' due to unexpected filename format.")
                continue

            file_path = os.path.join(input_folder, file)
            df = pd.read_csv(file_path)
        
            mean_values = df[['R', 'G', 'B', 'L', 'a', 'b']].mean()
            mean_values['time'] = time_int
        
            all_data.append(mean_values)
    
        # Create a DataFrame from the list of mean values
        result_df = pd.DataFrame(all_data)
        result_df = result_df.sort_values(by='time')

        # Save the average values to a new CSV file
        result_df.to_csv(output_file, index=False)
        print(f"Average color data saved to '{output_file}'.")

        return result_df

    def plot_color(self, df1, df2, df3, output_path):
        # Plot L vs Time
        plt.figure(figsize=(5,5))
        plt.scatter(df1['time'], df1['L'], marker='o', linestyle='-', label='60°C', color='black', facecolors='white', s=36)
        plt.scatter(df2['time'], df2['L'], marker='s', linestyle='-', label='70°C', color='black', facecolors='black', s=36)
        plt.scatter(df3['time'], df3['L'], marker='^', linestyle='-', label='80°C', color='black', facecolors='white', s=36)
        plt.xlabel("Time (min)", fontsize=12)
        plt.ylabel("L*", fontsize=12)
        # plt.title("L vs Time", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_path, "color_L_vs_Time.png"), dpi=300)
        plt.close()

        # Plot a* vs Time
        plt.figure(figsize=(5,5))
        plt.scatter(df1['time'], df1['a'], marker='o', linestyle='-', label='60°C', color='black', facecolors='white', s=36)
        plt.scatter(df2['time'], df2['a'], marker='s', linestyle='-', label='70°C', color='black', facecolors='black', s=36)
        plt.scatter(df3['time'], df3['a'], marker='^', linestyle='-', label='80°C', color='black', facecolors='white', s=36)
        plt.xlabel("Time (min)", fontsize=12)
        plt.ylabel("a*")
        # plt.title("a* vs Time", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_path, "color_a_vs_Time.png"), dpi=300)
        plt.close()
    
        # Plot b* vs Time
        plt.figure(figsize=(5,5))
        plt.scatter(df1['time'], df1['b'], marker='o', linestyle='-', label='60°C', color='black', facecolors='white', s=36)
        plt.scatter(df2['time'], df2['b'], marker='s', linestyle='-', label='70°C', color='black', facecolors='black', s=36)
        plt.scatter(df3['time'], df3['b'], marker='^', linestyle='-', label='80°C', color='black', facecolors='white', s=36)
        plt.xlabel("Time (min)", fontsize=12)
        plt.ylabel("b*")
        # plt.title("b* vs Time", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_path, "color_b_vs_Time.png"), dpi=300)
        plt.close()

        # Calculate colour difference (ΔEab)
        deltaE60 = np.sqrt((df1["L"] - df1["L"].iloc[0])**2 + (df1["a"] - df1["a"].iloc[0])**2 + (df1["b"] - df1["b"].iloc[0])**2)
        deltaE70 = np.sqrt((df2["L"] - df2["L"].iloc[0])**2 + (df2["a"] - df2["a"].iloc[0])**2 + (df2["b"] - df2["b"].iloc[0])**2)
        deltaE80 = np.sqrt((df3["L"] - df3["L"].iloc[0])**2 + (df3["a"] - df3["a"].iloc[0])**2 + (df3["b"] - df3["b"].iloc[0])**2)
        
        df1["deltaE"] = deltaE60
        df2["deltaE"] = deltaE70
        df3["deltaE"] = deltaE80
        
        # plot ΔEab
        plt.figure(figsize=(5,5))
        plt.scatter(df1["time"], df1["deltaE"], label="60°C", marker='o', linestyle='-', color='black', facecolors='white', s=36)
        plt.scatter(df2["time"], df2["deltaE"], label="70°C", marker='s', linestyle='-', color='black', facecolors='black', s=36)
        plt.scatter(df3["time"], df3["deltaE"], label="80°C", marker='^', linestyle='-', color='black', facecolors='white', s=36)
        plt.xlabel("Time (min)", fontsize=12)
        plt.ylabel("ΔE", fontsize=12)
        # plt.title("Colour Difference (ΔEab) at Different Temperatures", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_path, "color_deltaE_vs_Time.png"), dpi=300)
        plt.close()

        # Calculate Chroma (C)
        C60 = np.sqrt(df1["a"]**2 + df1["b"]**2)
        C70 = np.sqrt(df2["a"]**2 + df2["b"]**2)
        C80 = np.sqrt(df3["a"]**2 + df3["b"]**2)
        
        df1["C"] = C60
        df2["C"] = C70
        df3["C"] = C80
        
        # plot C
        plt.figure(figsize=(5,5))
        plt.scatter(df1["time"], df1["C"], label="60°C", marker='o', linestyle='-', color='black', facecolors='white', s=36)
        plt.scatter(df2["time"], df2["C"], label="70°C", marker='s', linestyle='-', color='black', facecolors='black', s=36)
        plt.scatter(df3["time"], df3["C"], label="80°C", marker='^', linestyle='-', color='black', facecolors='white', s=36)
        plt.xlabel("Time (min)", fontsize=12)
        plt.ylabel("C*", fontsize=12)
        # plt.title("Chroma (C*) at Different Temperatures", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_path, "color_C_vs_Time.png"), dpi=300)
        plt.close()

        # # Calculate Hue value
        h60 = np.rad2deg(np.arctan2(df1['b'], df1['a']))
        h70 = np.rad2deg(np.arctan2(df2['b'], df2['a']))
        h80 = np.rad2deg(np.arctan2(df3['b'], df3['a']))

        df1['h'] = h60
        df2['h'] = h70
        df3['h'] = h80

        # Plot hue
        plt.figure(figsize=(5,5))
        plt.scatter(df1["time"], df1["h"], label="60°C", marker='o', linestyle='-', color='black', facecolors='white', s=36)
        plt.scatter(df2["time"], df2["h"], label="70°C", marker='s', linestyle='-', color='black', facecolors='black', s=36)
        plt.scatter(df3["time"], df3["h"], label="80°C", marker='^', linestyle='-', color='black', facecolors='white', s=36)
        plt.xlabel("Time (min)", fontsize=12)
        plt.ylabel("h", fontsize=12)
        # plt.title("Hue (h) at Different Temperatures", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_path, "color_h_vs_Time.png"), dpi=300)
        plt.close()

        # Calculate total saturation difference, ΔC, ∆C^* = √((a^* )^2+(b^* )^2 ) - √((a_0^* )^2+(b_0^* )^2 )
        deltaC60 = np.sqrt((df1["a"]**2 + df1["b"]**2)) - np.sqrt((df1["a"].iloc[0]**2 + df1["b"].iloc[0]**2))
        deltaC70 = np.sqrt((df2["a"]**2 + df2["b"]**2)) - np.sqrt((df2["a"].iloc[0]**2 + df2["b"].iloc[0]**2))
        deltaC80 = np.sqrt((df3["a"]**2 + df3["b"]**2)) - np.sqrt((df3["a"].iloc[0]**2 + df3["b"].iloc[0]**2))
        
        df1["deltaC"] = deltaC60
        df2["deltaC"] = deltaC70
        df3["deltaC"] = deltaC80
        
        # Plot ΔC
        plt.figure(figsize=(5,5))
        plt.scatter(df1["time"], df1["deltaC"], label="60°C", marker='o', linestyle='-', color='black', facecolors='white', s=36)
        plt.scatter(df2["time"], df2["deltaC"], label="70°C", marker='s', linestyle='-', color='black', facecolors='black', s=36)
        plt.scatter(df3["time"], df3["deltaC"], label="80°C", marker='^', linestyle='-', color='black', facecolors='white', s=36)
        plt.xlabel("Time (min)", fontsize=12)
        plt.ylabel("ΔC*", fontsize=12)
        # plt.title("Total Saturation Difference (ΔC) at Different Temperatures", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_path, "color_deltaC_vs_Time.png"), dpi=300)
        plt.close()

        # Calculate total hue difference, Δh, ∆h=√((∆E)^2+(∆L^* )^2+(∆C^* )^2 )
        # ∆L^*=L^*-L_0^*
        deltaL60 = df1["L"] - df1["L"].iloc[0]
        deltaL70 = df2["L"] - df2["L"].iloc[0]
        deltaL80 = df3["L"] - df3["L"].iloc[0]
        deltaH60 = np.sqrt((df1["deltaE"]**2 + deltaL60**2 + df1["deltaC"]**2))
        deltaH70 = np.sqrt((df2["deltaE"]**2 + deltaL70**2 + df2["deltaC"]**2))
        deltaH80 = np.sqrt((df3["deltaE"]**2 + deltaL80**2 + df3["deltaC"]**2))
        
        df1["deltaH"] = deltaH60
        df2["deltaH"] = deltaH70
        df3["deltaH"] = deltaH80
        
        # Plot Δh
        plt.figure(figsize=(5,5))
        plt.scatter(df1["time"], df1["deltaH"], label="60°C", marker='o', linestyle='-', color='black', facecolors='white', s=36)
        plt.scatter(df2["time"], df2["deltaH"], label="70°C", marker='s', linestyle='-', color='black', facecolors='black', s=36)
        plt.scatter(df3["time"], df3["deltaH"], label="80°C", marker='^', linestyle='-', color='black', facecolors='white', s=36)
        plt.xlabel("Time (min)", fontsize=12)
        plt.ylabel("Δh", fontsize=12)
        # plt.title("Total Hue Difference (Δh) at Different Temperatures", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_path, "color_deltaH_vs_Time.png"), dpi=300)
        plt.close()

        # Browning index, BI=100×((X-0.31)/0.17), where X is obtained from X=((a^*+1.75∙L^* )  a^*)/((5.645∙L^*+a^*-3.012∙b^* ) )
        X60 = (1.75 * df1["L"] + df1["a"]) / (5.645 * df1["L"] + df1["a"] - 3.012 * df1["b"])
        X70 = (1.75 * df2["L"] + df2["a"]) / (5.645 * df2["L"] + df2["a"] - 3.012 * df2["b"])
        X80 = (1.75 * df3["L"] + df3["a"]) / (5.645 * df3["L"] + df3["a"] - 3.012 * df3["b"])
        browning60 = 100 * (X60 - 0.31) / 0.17
        browning70 = 100 * (X70 - 0.31) / 0.17
        browning80 = 100 * (X80 - 0.31) / 0.17
        
        df1["BI"] = browning60
        df2["BI"] = browning70
        df3["BI"] = browning80
        
        # Plot BI
        plt.figure(figsize=(5,5))
        plt.scatter(df1["time"], df1["BI"], label="60°C", marker='o', linestyle='-', color='black', facecolors='white', s=36)
        plt.scatter(df2["time"], df2["BI"], label="70°C", marker='s', linestyle='-', color='black', facecolors='black', s=36)
        plt.scatter(df3["time"], df3["BI"], label="80°C", marker='^', linestyle='-', color='black', facecolors='white', s=36)
        plt.xlabel("Time (min)", fontsize=12)
        plt.ylabel("BI", fontsize=12)
        # plt.title("Browning Index (BI) at Different Temperatures", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_path, "color_BI_vs_Time.png"), dpi=300)
        plt.close()
        
        # Plot redness index, RI = a / b
        RI60 = df1["a"] / df1["b"]
        RI70 = df2["a"] / df2["b"]
        RI80 = df3["a"] / df3["b"]
        
        df1["RI"] = RI60
        df2["RI"] = RI70
        df3["RI"] = RI80
        
        # Plot RI
        plt.figure(figsize=(5,5))
        plt.scatter(df1["time"], df1["RI"], label="60°C", marker='o', linestyle='-', color='black', facecolors='white', s=36)
        plt.scatter(df2["time"], df2["RI"], label="70°C", marker='s', linestyle='-', color='black', facecolors='black', s=36)
        plt.scatter(df3["time"], df3["RI"], label="80°C", marker='^', linestyle='-', color='black', facecolors='white', s=36)
        plt.xlabel("Time (min)", fontsize=12)
        plt.ylabel("RI", fontsize=12)
        # plt.title("Redness Index (RI) at Different Temperatures", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_path, "color_RI_vs_Time.png"), dpi=300)
        plt.close()
     
        # Plot yellowness index, YI = (142.86 * b) / L
        YI60 = (142.86 * df1["b"]) / df1["L"]
        YI70 = (142.86 * df2["b"]) / df2["L"]
        YI80 = (142.86 * df3["b"]) / df3["L"]
        
        df1["YI"] = YI60
        df2["YI"] = YI70
        df3["YI"] = YI80
        
        # Plot YI
        plt.figure(figsize=(5,5))
        plt.scatter(df1["time"], df1["YI"], label="60°C", marker='o', linestyle='-', color='black', facecolors='white', s=36)
        plt.scatter(df2["time"], df2["YI"], label="70°C", marker='s', linestyle='-', color='black', facecolors='black', s=36)
        plt.scatter(df3["time"], df3["YI"], label="80°C", marker='^', linestyle='-', color='black', facecolors='white', s=36)
        plt.xlabel("Time (min)", fontsize=12)
        plt.ylabel("YI", fontsize=12)
        # plt.title("Yellownes Index (YI) at Different Temperatures", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_path, "color_YI_vs_Time.png"), dpi=300)
        plt.close()

        # Save all data to CSV file
        df1.to_csv(os.path.join(output_path, "color_60C_updated.csv"), index=False)
        df2.to_csv(os.path.join(output_path, "color_70C_updated.csv"), index=False)
        df3.to_csv(os.path.join(output_path, "color_80C_updated.csv"), index=False)

        return df1, df2, df3

    def color(self, ds, temp, path, col):
        # Normalisasi terhadap nilai awal
        C0 = ds[col].iloc[0]
        ds[f"{col}_norm"] = ds[col] / C0

        # Save all data to CSV file
        ds.to_csv(os.path.join(path, f"color_{temp}C_updated.csv"), index=False)

        # Zero-order model
        def zero_order(t, c0, k):       
            return c0 - k*t
        # First-order model
        def first_order(t, c0, k):      
            return c0 * np.exp(-k*t)
        # Second-order model
        def second_order(t, c0, k):     
            return c0 / (1 + c0*k*t)
        # Weibull
        def weibull(t, c0, a, b):       
            return c0 * np.exp(-((t/a)**b))
        # Two-degree polinomial
        def two_degree_polynomial(t, a, b, c):    
            return a + b*t + c*t**2
        # Three-degree polinomial
        def three_degree_polynomial(t, a, b, c, d):    
            return a + b*t + c*t**2 + d*t**3
        # Wang-Sing MODE
        def wang_sing(t, a, b):      
            return 1 + a * t + b * t**2
        # Page model
        def page(t, n, k):    
            return np.exp(k*t**n)
        
        # Fit with initial guesses
        params_zo, _ = curve_fit(zero_order, ds["time"], ds[f"{col}_norm"], p0=(1,0.001))
        c0_zo, k_zo = params_zo; print("Zero-order parameters: c0 =", c0_zo, "k =", k_zo)
        params_fo, _ = curve_fit(first_order, ds["time"], ds[f"{col}_norm"], p0=(1,0.001), maxfev=10000)
        c0_fo, k_fo = params_fo; print("First-order parameters: c0 =", c0_fo, "k =", k_fo)
        params_so, _ = curve_fit(second_order, ds["time"], ds[f"{col}_norm"], p0=(1,0.001), maxfev=10000)
        c0_so, k_so = params_so; print("Second-order parameters: c0 =", c0_so, "k =", k_so)
        
        params_we, _ = curve_fit(weibull, ds["time"], ds[f"{col}_norm"], p0=(ds[f"{col}_norm"].iloc[0], 50, 1), bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=10000)
        c0_we, a_we, b_we = params_we; print("Weibull parameters: c0 =", c0_we, "a =", a_we, "b =", b_we)
        params_td, _ = curve_fit(two_degree_polynomial, ds["time"], ds[f"{col}_norm"], p0=(1,0.001,0.001))
        a_td, b_td, c_td = params_td; print("Two-degree polynomial parameters: a =", a_td, "b =", b_td, "c =", c_td)
        params_thd, _ = curve_fit(three_degree_polynomial, ds["time"], ds[f"{col}_norm"], p0=(1,0.001,0.001,0.001))
        a_thd, b_thd, c_thd, d_thd = params_thd; print("Three-degree polynomial parameters: a =", a_thd, "b =", b_thd, "c =", c_thd, "d =", d_thd)
        params_ws, _ = curve_fit(wang_sing, ds["time"], ds[f"{col}_norm"], p0=(1,0.001))
        a_ws, b_ws = params_ws; print("Wang-Sing model parameters: a =", a_ws, "b =", b_ws)
        
        params_po, _ = curve_fit(page, ds["time"], ds[f"{col}_norm"], p0=(1,0.001))
        n_po, k_po = params_po; print("Page model parameters: n =", n_po, "k =", k_po)
        # Print constants result on txt
        with open(f"{path}/results/{col}_{temp_exp}_constants.txt", "w") as f:
            f.write("Zero-order parameters: c0 = " + str(c0_zo) + " k = " + str(k_zo) + "\n")
            f.write("First-order parameters: c0 = " + str(c0_fo) + " k = " + str(k_fo) + "\n")
            f.write("Second-order parameters: c0 = " + str(c0_so) + " k = " + str(k_so) + "\n")
            f.write("Weibull parameters: c0 = " + str(c0_we) + " a = " + str(a_we) + " b = " + str(b_we) + "\n")
            f.write("Two-degree polynomial parameters: a = " + str(a_td) + " b = " + str(b_td) + " c = " + str(c_td) + "\n")
            f.write("Three-degree polynomial parameters: a = " + str(a_thd) + " b = " + str(b_thd) + " c = " + str(c_thd) + " d = " + str(d_thd) + "\n")
            f.write("Wang-Sing model parameters: a = " + str(a_ws) + " b = " + str(b_ws) + "\n")
            f.write("Page model parameters: n = " + str(n_po) + " k = " + str(k_po) + "\n")
        
        # Predictions
        ds[f"{col}_zo"] = zero_order(ds["time"], *params_zo)
        ds[f"{col}_fo"] = first_order(ds["time"], *params_fo)
        ds[f"{col}_so"] = second_order(ds["time"], *params_so)
        ds[f"{col}_we"] = weibull(ds["time"], *params_we)   
        ds[f"{col}_td"] = two_degree_polynomial(ds["time"], *params_td)
        ds[f"{col}_thd"] = three_degree_polynomial(ds["time"], *params_thd)
        ds[f"{col}_ws"] = wang_sing(ds["time"], *params_ws)
        ds[f"{col}_po"] = page(ds["time"], *params_po)
        
        # Goodness-of-fit function  
        def metrics(obs, pred):
            SS_res = np.sum((obs - pred)**2)
            SS_tot = np.sum((obs - np.mean(obs))**2)
            R2 = 1 - SS_res/SS_tot
            RMSE = np.sqrt(SS_res/len(obs))
            chi2 = np.sum(((obs - pred)**2)/(pred+1e-8))
            return R2, RMSE, chi2
        
        # Collect results
        results = {
            "Zero-order": metrics(ds[f"{col}_norm"], ds[f"{col}_zo"]),
            "First-order": metrics(ds[f"{col}_norm"], ds[f"{col}_fo"]),
            "Second-order": metrics(ds[f"{col}_norm"], ds[f"{col}_so"]),
            "Weibull": metrics(ds[f"{col}_norm"], ds[f"{col}_we"]),
            "Two-degree polynomial": metrics(ds[f"{col}_norm"], ds[f"{col}_td"]),
            "Three-degree polynomial": metrics(ds[f"{col}_norm"], ds[f"{col}_thd"]),
            "Wang-Sing": metrics(ds[f"{col}_norm"], ds[f"{col}_ws"]),
            "Page": metrics(ds[f"{col}_norm"], ds[f"{col}_po"]),
            }
        # Print table
        print(f"{'Model':<20}{'R²':<15}{'RMSE':<15}{'Chi²':<15}")
        for model, (R2, RMSE, chi2) in results.items(): 
            print(f"{model:<20}{R2:<15.4f}{RMSE:<15.4f}{chi2:<15.4f}")
        # Save results of R2, RMSE, and chi-square to CSV
        res_df = pd.DataFrame(results).T
        res_df.columns = ["R²", "RMSE", "Chi²"]
        res_df.to_csv(f"{path}/results/{col}_{temp_exp}_fit.csv", index=True)
        # Plot observed data
        plt.figure(figsize=(6,5))
        plt.scatter(ds["time"], ds[f"{col}_norm"], color="black", label="Observed Color", s=25)

        # Plot predictions for each model
        plt.plot(ds["time"], ds[f"{col}_zo"], label="Zero-order", lw = 2.0)
        plt.plot(ds["time"], ds[f"{col}_fo"], label="First-order", lw = 2.0)
        plt.plot(ds["time"], ds[f"{col}_so"], label="Second-order", lw = 2.0)
        plt.plot(ds["time"], ds[f"{col}_we"], label="Weibull", lw = 2.0)
        plt.plot(ds["time"], ds[f"{col}_td"], label="Two-degree polynomial", lw = 2.0)
        plt.plot(ds["time"], ds[f"{col}_thd"], label="Three-degree polynomial", lw = 2.0)
        plt.plot(ds["time"], ds[f"{col}_ws"], label="Wang-Sing", lw = 2.0)
        plt.plot(ds["time"], ds[f"{col}_po"], label="Page", lw = 2.0)
        plt.xlabel("Time (min)", fontsize=11)
        plt.ylabel(col, fontsize=11)
        plt.title(f"Observed vs Predicted {col} for Drying Models")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{path}/results/{col}_{temp_exp}_pred.png")

        return ds

    def computeEa(self, ds, row_label):
        """
        Compute activation energy (Ea) for a given row in the dataset.
        ds: pandas DataFrame with columns ['60','70','80'] and index ['L','a','b','C','BI','YI']
        row_label: string, e.g. 'a', 'L', 'b'
        """
        # Extract k values for the row
        k_values = ds.loc[row_label, ["temp_60", "temp_70", "temp_80"]].values.astype(float)

        # Temperatures in Kelvin
        T = np.array([60, 70, 80]) + 273.15
        R = 8.314  # J/(mol·K)

        # Linear regression ln(k) vs 1/T
        slope, intercept, _, _, _ = linregress(1/T, np.log(k_values))

        k0 = np.exp(intercept)
        Ea = -slope * R
        Ea_kj = Ea / 1000  # convert to kJ/mol

        print(f"Row: {row_label}")
        print(f"  Pre-exponential factor (k0): {k0:.6e}")
        print(f"  Activation Energy (Ea): {Ea_kj:.3f} kJ/mol")
        print(f"  Equation: ln(k) = ln({k0:.6e}) - ({Ea:.3f}/{R})(1/T)")
        
        # R-squared
        # R-squared
        ss_res = np.sum((np.log(k_values) - (slope * (1/T) + intercept))**2)
        ss_tot = np.sum((np.log(k_values) - np.mean(np.log(k_values)))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"R-squared: {r_squared}")


        return k0, Ea_kj, r_squared

if __name__ == "__main__":   

    """IDENTIFY SAMPLES, INITIAL MOISTURE AND TEMPERATURE"""
    sample = 'cmb';  temp_exp = 80;  init_moisture = 0.86611
    dataset = f"d://z/master/RaspberryPi/program/dataset/"
    
    """RUN PROGRAM"""
    model = DryingKinetics()

    """KINETICS ON MOISTURE LOSS""" 
    # input_moisture = dataset + f"moisture/{sample}_{temp_exp}_MCLoss.csv"
    # output_moisture = dataset + f"moisture/{sample}_{temp_exp}_MR.csv"

    # ds_MR = model.load_data(input_moisture)
    # ds_MR = model.computeMR(ds_MR, temp_exp, init_moisture, output_moisture)
    # ds_MR = model.moisture(ds_MR, temp_exp, output_moisture)

    """PLOT MOISTURE RATIO AND DRYING RATE AT EACH TEMPERATURE"""
    # ds_60 = model.load_data(f"{dataset}moisture/cmb_60_MR.csv_updated.csv")
    # ds_70 = model.load_data(f"{dataset}moisture/cmb_70_MR.csv_updated.csv")
    # ds_80 = model.load_data(f"{dataset}moisture/cmb_80_MR.csv_updated.csv")
    # out1 = f"{dataset}moisture/cmb_MCLoss_MRxtime_plot.png"
    # out2 = f"{dataset}moisture/cmb_MCLoss_DRxMR_plot.png"
    # out3 = f"{dataset}moisture/cmb_MCLoss_DRxtime_plot.png"
    # model.plot_MRxDR(ds_60, ds_70, ds_80, out1, out2, out3)

    """DRYING CHARACTERISTICS"""
    # output = f"{dataset}moisture/cmb_MCLoss_Ea.png"
    # model.MCDryAttributes(ds_60, ds_70, ds_80, output)

    """FIND AVERAGE CIELab values""" 
    # input_color = dataset + f"color/{sample}_{temp_exp}_color/"
    # output_color = dataset + f"color/{sample}_{temp_exp}_color.csv"
    # df_color = model.average_Lab(input_color, output_color)

    """PLOT CIELab VALUES"""
    # ds_60 = model.load_data(f"{dataset}color/cmb_60_color_.csv")
    # ds_70 = model.load_data(f"{dataset}color/cmb_70_color_.csv")
    # ds_80 = model.load_data(f"{dataset}color/cmb_80_color_.csv")
    # out_path = f"{dataset}color/"
    # df_60, df_70, df_80 = model.plot_color(ds_60, ds_70, ds_80, out_path)

    """KINETICS ON COLOR DEGRADATION"""
    input_Lab = dataset + f"color/color_{temp_exp}C_updated.csv"
    output_Lab = dataset + f"color/"
    ds_color = model.load_data(input_Lab)
    ds_color = model.color(ds_color, temp_exp, output_Lab, col='L')
    ds_color = model.color(ds_color, temp_exp, output_Lab, col='a')
    ds_color = model.color(ds_color, temp_exp, output_Lab, col='b')
    # ds_color = model.color(ds_color, temp_exp, output_Lab, col="deltaE")
    ds_color = model.color(ds_color, temp_exp, output_Lab, col='C')
    # ds_color = model.color(ds_color, temp_exp, output_Lab, col='h')
    # ds_color = model.color(ds_color, temp_exp, output_Lab, col="deltaC")
    # ds_color = model.color(ds_color, temp_exp, output_Lab, col="deltaH") 
    ds_color = model.color(ds_color, temp_exp, output_Lab, col='BI')
    ds_color = model.color(ds_color, temp_exp, output_Lab, col='YI')

    # # input_k = dataset + f"color/cmb_k.csv"
    # # read_k = pd.read_csv(input_k)
    # # read_k.columns = ["clr", "temp_60", "temp_70", "temp_80"]
    # # read_k = read_k.set_index("clr")
    # # # Loop through all rows and compute Do and Ea
    # # results = {}
    # # for row in read_k.index:
    # #     Do, Ea_kj, r_squared = model.computeEa(read_k, row)
    # #     results[row] = [Do, Ea_kj, r_squared]
    # # print("Summary of Do and Ea values (kJ/mol):")
    # # print(results)
    # # # Save in a excel file
    # # results_df = pd.DataFrame(results).T
    # # results_df.columns = ["Do", "Ea (kJ/mol)", "R-squared"]
    # # results_df.to_excel(f"{dataset}color/cmb_k_Do_Ea.xlsx", index=True)


