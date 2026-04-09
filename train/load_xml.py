import sys
import os
import xml.etree.ElementTree as ET
import traceback

import mujoco
import matplotlib

def load_mujoco_model(xml_path):
    """Intenta cargar un modelo de MuJoCo desde un archivo XML"""
    try:        
        print(f"\n{'='*60}")
        print(f"Intentando cargar: {xml_path}")
        print(f"Versión de MuJoCo: {mujoco.__version__}")
        print(f"{'='*60}\n")
        
        # Intentar cargar el modelo
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        
        print("✓ MODELO CARGADO EXITOSAMENTE")
        print(f"  - nq: {model.nq} (posiciones)")
        print(f"  - nv: {model.nv} (velocidades)")
        print(f"  - nu: {model.nu} (actuadores)")
        print(f"  - nbody: {model.nbody} (cuerpos)")
        print(f"  - ngeom: {model.ngeom} (geometrías)")
        print(f"  - njnt: {model.njnt} (articulaciones)")
        
        return model, data
        
    except Exception as e:
        print(f"✗ ERROR AL CARGAR EL MODELO:")
        print(f"  {type(e).__name__}: {e}")
        
        # Intentar obtener más detalles
        if hasattr(e, 'args') and len(e.args) > 0:
            print(f"\nDetalles adicionales:")
            for arg in e.args:
                print(f"  - {arg}")
        
        # Mostrar traceback completo si se desea
        if '--verbose' in sys.argv or '-v' in sys.argv:
            print("\nTraceback completo:")
            traceback.print_exc()
        
        return None, None
    

if __name__ == "__main__":

    xml = "/root/EL7009_projects/mujoco_playground_el7009_project/mujoco_playground/_src/locomotion"
    robot = "go2/xmls/go2_mjx_feetonly.xml"

    file = f"{xml}/{robot}"
    #load_mujoco_model(file)
    print(matplotlib.get_backend())