import numpy as np, glob, os         
out_dir = '/home/lukau/Desktop/point2cad_repr/output_brep/00949'                              
for p in sorted(glob.glob(os.path.join(out_dir, 'inter_*.npz')))[:3]:
    d = np.load(p, allow_pickle=True)
    print(os.path.basename(p), 'type=', str(d['curve_type']), 'n_curves=', int(d['n_curves']))
    for k in range(int(d['n_curves'])):          
        pts = d[f'curve_points_{k}']          
        print(f'  curve_points_{k}: shape={pts.shape}  min={pts.min(0)}  max={pts.max(0)}')          
    for k in range(int(d['n_untrimmed_curves'])):
        pts = d[f'untrimmed_curve_points_{k}']
        print(f'  untrimmed_curve_points_{k}: shape={pts.shape}  min={pts.min(0)}  max={pts.max(0)}')