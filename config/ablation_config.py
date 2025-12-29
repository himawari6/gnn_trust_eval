from dataclasses import dataclass

@dataclass
class AblationConfig:
    mode: str
    use_vm2user_edge: bool
    tag: str


ABLATION_CONFIGS = {
    'U': AblationConfig(
        mode='U',
        use_vm2user_edge=False,
        tag='UserOnly'
    ),
    'UT': AblationConfig(
        mode='UT',
        use_vm2user_edge=False,
        tag='UserTerminal'
    ),
    'UV': AblationConfig(
        mode='UV',
        use_vm2user_edge=True,
        tag='UserVM'
    ),
    'UTV': AblationConfig(
        mode='UTV',
        use_vm2user_edge=False,
        tag='FullGraph'
    )
}
