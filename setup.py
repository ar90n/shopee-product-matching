from setuptools import find_packages, setup


def main(**kwargs):
    setup(
        name="shopee-product-matching",
        version="2.0.0",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        data_files=[],
        entry_points={
            "console_scripts": [
                "shopee_product_matching=shopee_product_matching.cli:main"
            ]
        },
        **kwargs
    )


if __name__ == "__main__":
    main()
