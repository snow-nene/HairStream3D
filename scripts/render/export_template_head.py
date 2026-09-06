"""导出 Blender 模板最终求值头模到发丝 Y-up 世界坐标。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    obj=bpy.data.objects['head_smooth']
    # 与渲染一致地启用修改器的 render 细分级别。
    modifiers=[]
    for modifier in obj.modifiers:
        modifiers.append({'name':modifier.name,'type':modifier.type,
                          'show_render':modifier.show_render})
        modifier.show_viewport=modifier.show_render
        if modifier.type=='SUBSURF':modifier.levels=modifier.render_levels
    bpy.context.view_layer.update()
    evaluated=obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh=evaluated.to_mesh();mesh.calc_loop_triangles()
    vertices=np.array([list(evaluated.matrix_world@v.co) for v in mesh.vertices])
    vertices=vertices[:,[0,2,1]]*np.array([1,1,-1])
    faces=np.array([list(t.vertices) for t in mesh.loop_triangles])
    np.savez_compressed(args.output_dir/'template_head.npz',vertices=vertices,faces=faces)
    report={'template':bpy.data.filepath,'sha256':hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
            'object':obj.name,'modifiers':modifiers,'vertices':len(vertices),'faces':len(faces),
            'coordinates':'world metres Y-up; inverse of hair (x,y,z)->(x,-z,y)',
            'bounds':[vertices.min(0).tolist(),vertices.max(0).tolist()]}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True);evaluated.to_mesh_clear()


if __name__=='__main__':main()
