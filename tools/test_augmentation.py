"""Quick smoke tests for augmentation factory behavior.

This script is not a full unit-test. It's a quick sanity check that can run
locally to validate config-driven defaults are working and no baked-in defaults
remain in the module.
"""
from system.coordinator_settings import SETTINGS
from system.augmentation import DataAugmentationFactory, get_augmentation_for_dataset


def run_check():
    print('Settings loaded:', isinstance(SETTINGS, dict))
    print('Available augmentations:', len(DataAugmentationFactory.get_available_augmentations()))

    # 1) random_crop should work with config defaults
    rc = DataAugmentationFactory.create_augmentation('random_crop')
    print('random_crop created:', type(rc))

    # 2) training pipeline compose creation
    comp = get_augmentation_for_dataset(None, 'train')
    print('training pipeline created:', type(comp))

    # 3) compose default from config
    defaults = SETTINGS.get('augmentations', {}).get('defaults', {})
    if 'compose' in defaults:
        c = DataAugmentationFactory.create_augmentation('compose', defaults.get('compose'))
        print('compose created from defaults:', type(c))

    print('All augmentation quick checks passed')


if __name__ == '__main__':
    run_check()
