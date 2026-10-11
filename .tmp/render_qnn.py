import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

root = Path(__file__).resolve().parents[1]
notebook_path = root / 'quantum_neural_networks_intro.ipynb'
notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
source = '''# Draw every operation in the trained three-layer circuit.
import importlib
import pennylane.drawer.mpldrawer as mpldrawer

# Refresh PennyLane's cached optional import if Matplotlib was installed
# after PennyLane was first imported in this notebook session.
if not mpldrawer.has_mpl:
    importlib.reload(mpldrawer)

fig = plt.figure(figsize=(28, 5.2))
fig, ax = qml.draw_mpl(
    quantum_circuit, fig=fig, decimals=1, level='device',
    wire_order=WIRES, show_all_wires=True,
)(X_train_qnn[0], qnn_best_parameters[0])
ax.set_title('Complete trained Iris QNN: four qubits and three trainable layers', pad=18)
fig.savefig('qnn_iris_circuit.png', dpi=180, bbox_inches='tight')
fig.savefig('qnn_iris_circuit.svg', bbox_inches='tight')
plt.show()
'''
namespace = {}
for index in (2, 4, 6, 8):
    exec(''.join(notebook['cells'][index]['source']), namespace)
model = json.loads((root / 'qnn_iris_model.json').read_text(encoding='utf-8'))
namespace['qnn_best_parameters'] = tuple(
    namespace['pnp'].array(model[key]) for key in ('theta', 'W', 'b')
)
exec(source.replace('plt.show()', 'plt.close(fig)'), namespace)
cell = notebook['cells'][14]
cell['source'] = source.splitlines(keepends=True)
cell['execution_count'] = None
cell['outputs'] = [{
    'output_type': 'display_data',
    'data': {
        'image/png': base64.b64encode((root / 'qnn_iris_circuit.png').read_bytes()).decode('ascii'),
        'text/plain': ['<Figure: complete trained four-qubit, three-layer Iris QNN>'],
    },
    'metadata': {},
}]
notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print('Rendered circuit and updated notebook output using saved trained parameters.')
