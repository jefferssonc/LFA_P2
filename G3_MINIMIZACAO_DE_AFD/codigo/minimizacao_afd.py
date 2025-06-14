import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch

# --------------------------- Definição do AFD Original ---------------------------
states_orig = ['A', 'B', 'C', 'D', 'E', 'F']
alphabet = ['0', '1']
transitions_orig = {
    ('A', '0'): 'B',
    ('A', '1'): 'C',
    ('B', '0'): 'D',
    ('B', '1'): 'E',
    ('C', '0'): 'D',
    ('C', '1'): 'E',
    ('D', '0'): 'D',
    ('D', '1'): 'D',
    ('E', '0'): 'E',
    ('E', '1'): 'E',
    ('F', '0'): 'F',
    ('F', '1'): 'F',
}
initial_state = 'A'
final_states = ['E']  # Apenas o estado "E" é final

# Constrói o grafo original usando NetworkX
G_orig = nx.DiGraph()
G_orig.add_nodes_from(states_orig)
for (u, sym), v in transitions_orig.items():
    G_orig.add_edge(u, v, label=sym)

# Calcula posições para o layout original (para interpolar depois)
pos_orig = nx.spring_layout(G_orig, seed=42)

# --------------------------- Definição do AFD Minimizado ---------------------------
# Após remoção de F (inacessível) e fusão dos estados B e C em B_C, temos:
states_final = ['A', 'B_C', 'D', 'E']
transitions_min = {
    ('A', '0'): 'B_C',  # A,0: originalmente ia para B, agora B e C fundem-se
    ('A', '1'): 'B_C',  # A,1: originalmente ia para C
    ('B_C', '0'): 'D',
    ('B_C', '1'): 'E',
    ('D', '0'): 'D',
    ('D', '1'): 'D',
    ('E', '0'): 'E',
    ('E', '1'): 'E',
}
G_final = nx.DiGraph()
G_final.add_nodes_from(states_final)
for (u, sym), v in transitions_min.items():
    G_final.add_edge(u, v, label=sym)

# Layout final para o AFD minimizado
pos_final = nx.spring_layout(G_final, seed=42)

# --------------------------- Configuração da Animação ---------------------------
# Dividindo a animação em segmentos:
# 0–20: AFD original completo
# 21–50: Fade-out do estado F
# 51–60: Exibe AFD sem F
# 61–100: Interpolação para fusão de B e C em B_C (merge)
# 101–120: AFD minimizado final

frame_total = 120


def lerp(pos1, pos2, t):
    """Interpolação linear entre dois pontos (x, y)."""
    return (pos1[0] * (1 - t) + pos2[0] * t, pos1[1] * (1 - t) + pos2[1] * t)


# Configuração da figura e eixo
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')


def draw_node(ax, pos, label, is_final=False, alpha=1.0, radius=0.08):
    """Desenha um nó com rótulo; se is_final True, marca com contorno duplo."""
    circ = Circle(pos, radius, facecolor='lightblue', edgecolor='black', lw=2, alpha=alpha, zorder=2)
    ax.add_patch(circ)
    if is_final:
        outer = Circle(pos, radius * 1.2, facecolor='none', edgecolor='black', lw=2, alpha=alpha, zorder=2)
        ax.add_patch(outer)
    ax.text(pos[0], pos[1], label, fontsize=12, ha='center', va='center', zorder=3, alpha=alpha)


def draw_arrow(ax, start, end, text, alpha=1.0):
    """Desenha uma seta entre dois pontos e adiciona um rótulo."""
    if np.allclose(start, end):
        # Para laços, desenha um pequeno círculo de volta para o nó
        loop = Circle((start[0], start[1] + 0.12), 0.1, fill=False, edgecolor='black', lw=1.5, alpha=alpha, zorder=1)
        ax.add_patch(loop)
        ax.text(start[0], start[1] + 0.22, text, fontsize=10, ha='center', va='center', color='red', alpha=alpha,
                zorder=3)
    else:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=20,
                                color='black', lw=1.5, alpha=alpha, zorder=1)
        ax.add_patch(arrow)
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(mid[0], mid[1], text, fontsize=10, ha='center', va='center', color='red', alpha=alpha, zorder=3)


def update(frame):
    ax.clear()
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')

    # Dicionários para armazenar os nós e arestas a serem desenhados
    nodes_to_draw = {}
    edges_to_draw = []  # cada item: (origem, destino, rótulo, alpha)

    if frame <= 20:
        # --- Segmento 1: AFD original completo ---
        for n in states_orig:
            nodes_to_draw[n] = (pos_orig[n], 1.0)
        for (u, sym), v in transitions_orig.items():
            edges_to_draw.append((u, v, sym, 1.0))

    elif frame <= 50:
        # --- Segmento 2: Fade-out do nó F ---
        for n in states_orig:
            if n == 'F':
                a = 1.0 if frame < 21 else max(0, 1 - (frame - 21) / (50 - 21))
                nodes_to_draw[n] = (pos_orig[n], a)
            else:
                nodes_to_draw[n] = (pos_orig[n], 1.0)
        for (u, sym), v in transitions_orig.items():
            a = 1.0
            if u == 'F' or v == 'F':
                a = 1.0 if frame < 21 else max(0, 1 - (frame - 21) / (50 - 21))
            edges_to_draw.append((u, v, sym, a))

    elif frame <= 60:
        # --- Segmento 3: AFD com F removido ---
        for n in states_orig:
            if n != 'F':
                nodes_to_draw[n] = (pos_orig[n], 1.0)
        for (u, sym), v in transitions_orig.items():
            if u != 'F' and v != 'F':
                edges_to_draw.append((u, v, sym, 1.0))

    elif frame <= 100:
        # --- Segmento 4: Fusão de B e C em B_C ---
        t = (frame - 61) / (100 - 61)  # t varia de 0 a 1
        # Para os nós que não são mesclados (A, D, E)
        for n in ['A', 'D', 'E']:
            pos_interp = lerp(pos_orig[n], pos_final[n], t)
            nodes_to_draw[n] = (pos_interp, 1.0)
        # Para B e C, ambos convergem para a posição de 'B_C'
        pos_B = lerp(pos_orig['B'], pos_final['B_C'], t)
        pos_C = lerp(pos_orig['C'], pos_final['B_C'], t)
        nodes_to_draw['B'] = (pos_B, 1.0)
        # O nó C desaparece gradualmente, evidenciando a fusão:
        alpha_C = max(0, 1 - t)
        nodes_to_draw['C'] = (pos_C, alpha_C)

        # Arestas: mantendo as transições originais (com fade para as de C)
        edges_to_draw.append(('A', 'B', '0', 1.0))
        edges_to_draw.append(('A', 'C', '1', 1.0))
        edges_to_draw.append(('B', 'D', '0', 1.0))
        edges_to_draw.append(('B', 'E', '1', 1.0))
        edges_to_draw.append(('C', 'D', '0', alpha_C))
        edges_to_draw.append(('C', 'E', '1', alpha_C))
        edges_to_draw.append(('D', 'D', '0/1', 1.0))
        edges_to_draw.append(('E', 'E', '0/1', 1.0))

    else:
        # --- Segmento 5: AFD minimizado final ---
        for n in states_final:
            nodes_to_draw[n] = (pos_final[n], 1.0)
        # Para a aresta de A, combinamos os rótulos 0 e 1 para o novo estado B_C
        edges_to_draw.append(('A', 'B_C', '0,1', 1.0))
        edges_to_draw.append(('B_C', 'D', '0', 1.0))
        edges_to_draw.append(('B_C', 'E', '1', 1.0))
        edges_to_draw.append(('D', 'D', '0/1', 1.0))
        edges_to_draw.append(('E', 'E', '0/1', 1.0))

    # Desenha as arestas primeiro para que os nós fiquem sobre elas
    for (u, v, sym, a) in edges_to_draw:
        if u in nodes_to_draw and v in nodes_to_draw:
            pos_u, alpha_u = nodes_to_draw[u]
            pos_v, alpha_v = nodes_to_draw[v]
            alpha_edge = a * (alpha_u + alpha_v) / 2
            draw_arrow(ax, pos_u, pos_v, sym, alpha=alpha_edge)

    # Desenha os nós
    for n, (p, a) in nodes_to_draw.items():
        is_final = (n == 'E')
        draw_node(ax, p, n, is_final=is_final, alpha=a)

    # Desenha a seta do estado inicial
    if frame <= 60:
        init_pos = pos_orig[initial_state]
    elif frame <= 100:
        init_pos = lerp(pos_orig[initial_state], pos_final[initial_state], (frame - 61) / (100 - 61))
    else:
        init_pos = pos_final[initial_state]
    start_init = (init_pos[0] - 0.3, init_pos[1])
    draw_arrow(ax, start_init, init_pos, "", alpha=1.0)


def main():
    ani = FuncAnimation(fig, update, frames=frame_total + 1, interval=100)
    plt.show()


if __name__ == '__main__':
    main()
