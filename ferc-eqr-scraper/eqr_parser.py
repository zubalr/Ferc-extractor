"""Custom parser for FERC EQR XML files (native format, not XBRL).

This parser handles FERC EQR XML filings which can contain:
- Multiple organizations (sellers, buyers, filers)
- Multiple contracts with products and transactions
- Contact information for organizations
- Large files (70MB+) with 400+ organizations and contracts

The parser uses efficient memory management for large files and proper
namespace handling for XML elements.
"""

import xml.etree.ElementTree as ET
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os


class EQRXMLParser:
    """Parser for FERC EQR XML files in native format.
    
    Handles large XML files efficiently with proper memory management
    and namespace support for the FERC EQR XML schema.
    """
    
    def __init__(self):
        """Initialize the EQR XML parser."""
        self.logger = logging.getLogger("ferc_scraper.eqr_parser")
        self.namespace = {"eqr": "urn:www.ferc.gov:forms.eqr"}
        
        # Track parsing statistics
        self.stats = {
            'files_processed': 0,
            'organizations_parsed': 0,
            'contacts_parsed': 0,
            'contracts_parsed': 0,
            'products_parsed': 0,
            'transactions_parsed': 0
        }
    
    def parse_xml_file(self, xml_path: str) -> Dict[str, pd.DataFrame]:
        """Parse a single EQR XML file into DataFrames.
        
        Efficiently handles large XML files with hundreds of organizations
        and contracts by using streaming-like parsing and memory management.
        
        Args:
            xml_path: Path to the XML file
            
        Returns:
            Dictionary of DataFrames with parsed data:
            - 'organizations': Organization/company data
            - 'contacts': Contact information for organizations  
            - 'contracts': Contract data
            - 'contract_products': Products associated with contracts
            - 'transactions': Transaction data
            
        Raises:
            ET.ParseError: If XML parsing fails
            Exception: For other parsing errors
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Parsing EQR XML file: {os.path.basename(xml_path)}")
            
            # Parse the XML file
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract filing metadata
            filing_metadata = self._extract_filing_metadata(root)
            self.logger.debug(f"Filing metadata: {filing_metadata}")
            
            # Parse different sections
            dataframes = {}
            
            # Parse organizations (includes sellers, buyers, filers)
            organizations_df = self._parse_organizations(root, filing_metadata)
            if not organizations_df.empty:
                dataframes['organizations'] = organizations_df
                self.stats['organizations_parsed'] += len(organizations_df)
            
            # Parse contacts
            contacts_df = self._parse_contacts(root, filing_metadata)
            if not contacts_df.empty:
                dataframes['contacts'] = contacts_df
                self.stats['contacts_parsed'] += len(contacts_df)
            
            # Parse contracts
            contracts_df = self._parse_contracts(root, filing_metadata)
            if not contracts_df.empty:
                dataframes['contracts'] = contracts_df
                self.stats['contracts_parsed'] += len(contracts_df)
            
            # Parse contract products
            contract_products_df = self._parse_contract_products(root, filing_metadata)
            if not contract_products_df.empty:
                dataframes['contract_products'] = contract_products_df
                self.stats['products_parsed'] += len(contract_products_df)
            
            # Parse transactions
            transactions_df = self._parse_transactions(root, filing_metadata)
            if not transactions_df.empty:
                dataframes['transactions'] = transactions_df
                self.stats['transactions_parsed'] += len(transactions_df)
            
            # Log results
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Successfully parsed {len(dataframes)} tables from XML file in {processing_time:.2f}s")
            
            for table_name, df in dataframes.items():
                self.logger.info(f"  {table_name}: {len(df)} rows")
            
            self.stats['files_processed'] += 1
            return dataframes
            
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error in {xml_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error parsing {xml_path}: {e}")
            raise
    
    def _extract_filing_metadata(self, root: ET.Element) -> Dict[str, Any]:
        """Extract filing-level metadata.
        
        Args:
            root: Root XML element
            
        Returns:
            Dictionary with filing metadata
        """
        metadata = {
            'filing_uid': root.get('Uid'),
            'period_type': root.get('PeriodType'),
            'year': root.get('Year'),
            'quarter': root.get('Quarter'),
            'filing_type': root.get('Type')
        }
        
        self.logger.debug(f"Filing metadata: {metadata}")
        return metadata
    
    def _parse_companies(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse companies (organizations) from the XML.
        
        This method is kept for backward compatibility but delegates to
        the new _parse_organizations method.
        
        Args:
            root: Root XML element
            filing_metadata: Filing metadata dictionary
            
        Returns:
            DataFrame with company/organization data
        """
        return self._parse_organizations(root, filing_metadata)
    
    def _parse_organizations(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse organizations (companies/entities) from the XML.
        
        Organizations can be filers, buyers, sellers, or combinations thereof.
        Each organization has a unique UID and may have a CID (Company ID).
        
        Args:
            root: Root XML element
            filing_metadata: Filing metadata dictionary
            
        Returns:
            DataFrame with organization data
        """
        organizations = []
        
        # Parse organizations using proper namespace
        org_elements = root.findall('eqr:Organizations/eqr:Organization', self.namespace)
        
        for org in org_elements:
            org_data = {
                'organization_uid': org.get('Uid'),
                'cid': org.get('Cid'),  # Company ID (may be None)
                'company_name': org.get('Name'),
                'is_filer': org.get('IsFiler') == 'true',
                'is_buyer': org.get('IsBuyer') == 'true',
                'is_seller': org.get('IsSeller') == 'true',
                'transactions_reported_to_index_publisher': org.get('TransactionsReportedToIndexPublisher') == 'true',
                'filing_uid': filing_metadata['filing_uid'],
                'year': filing_metadata['year'],
                'quarter': filing_metadata['quarter'],
                'period_type': filing_metadata['period_type'],
                'filing_type': filing_metadata['filing_type']
            }
            organizations.append(org_data)
        
        df = pd.DataFrame(organizations)
        if not df.empty:
            self.logger.debug(f"Parsed {len(df)} organizations")
        
        return df
    
    def _parse_contacts(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse contacts from organizations in the XML.
        
        Each organization can have multiple contacts with different roles
        (filer contact, buyer contact, seller contact).
        
        Args:
            root: Root XML element
            filing_metadata: Filing metadata dictionary
            
        Returns:
            DataFrame with contact data including addresses
        """
        contacts = []
        
        # Parse contacts from organizations using proper namespace
        organizations = root.findall('eqr:Organizations/eqr:Organization', self.namespace)
        for org in organizations:
            contact_elements = org.findall('eqr:Contacts/eqr:Contact', self.namespace)
            for contact in contact_elements:
                contact_data = {
                    'contact_uid': contact.get('Uid'),
                    'organization_uid': org.get('Uid'),
                    'first_name': contact.get('FirstName'),
                    'last_name': contact.get('LastName'),
                    'display_name': contact.get('DisplayName'),
                    'title': contact.get('Title'),
                    'phone': contact.get('Phone'),
                    'email': contact.get('Email'),
                    'is_filer_contact': contact.get('IsFilerContact') == 'true',
                    'is_buyer_contact': contact.get('IsBuyerContact') == 'true',
                    'is_seller_contact': contact.get('IsSellerContact') == 'true',
                    'filing_uid': filing_metadata['filing_uid'],
                    'year': filing_metadata['year'],
                    'quarter': filing_metadata['quarter'],
                    'period_type': filing_metadata['period_type'],
                    'filing_type': filing_metadata['filing_type']
                }
                
                # Parse address using namespace
                address = contact.find('eqr:Address', self.namespace)
                if address is not None:
                    contact_data.update({
                        'street1': address.get('Street1'),
                        'street2': address.get('Street2'),
                        'street3': address.get('Street3'),
                        'city': address.get('City'),
                        'state': address.get('State'),
                        'zip': address.get('Zip'),
                        'country': address.get('Country')
                    })
                
                contacts.append(contact_data)
        
        df = pd.DataFrame(contacts)
        if not df.empty:
            self.logger.debug(f"Parsed {len(df)} contacts")
        
        return df
    
    def _parse_contracts(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse contracts from the XML.
        
        Contracts link sellers and buyers and contain products and transactions.
        Each contract has execution dates, termination dates, and tariff references.
        
        Args:
            root: Root XML element
            filing_metadata: Filing metadata dictionary
            
        Returns:
            DataFrame with contract data
        """
        contracts = []
        
        # Parse contracts using proper namespace
        contract_elements = root.findall('eqr:Contracts/eqr:Contract', self.namespace)
        for contract in contract_elements:
            contract_data = {
                'contract_uid': contract.get('Uid'),
                'seller_uid': contract.get('SellerUid'),
                'buyer_uid': contract.get('BuyerUid'),
                'ferc_tariff_reference': contract.get('FercTariffReference'),
                'contract_service_agreement': contract.get('ContractServiceAgreement'),
                'is_affiliate': contract.get('IsAffiliate') == 'true',
                'execution_date': contract.get('ExecutionDate'),
                'commencement_date': contract.get('CommencementDate'),
                'termination_date': contract.get('TerminationDate'),
                'extension_provision_description': contract.get('ExtensionProvisionDescription'),
                'filing_type_contract': contract.get('FilingType'),  # Contract-level filing type
                'filing_uid': filing_metadata['filing_uid'],
                'year': filing_metadata['year'],
                'quarter': filing_metadata['quarter'],
                'period_type': filing_metadata['period_type'],
                'filing_type': filing_metadata['filing_type']  # Filing-level filing type
            }
            contracts.append(contract_data)
        
        df = pd.DataFrame(contracts)
        if not df.empty:
            self.logger.debug(f"Parsed {len(df)} contracts")
            
            # Convert date columns
            date_columns = ['execution_date', 'commencement_date', 'termination_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _parse_contract_products(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse contract products from the XML.
        
        Contract products define what is being traded under each contract,
        including capacity, energy, or other products with rates and terms.
        
        Args:
            root: Root XML element
            filing_metadata: Filing metadata dictionary
            
        Returns:
            DataFrame with contract product data
        """
        products = []
        
        # More efficient approach: iterate through contracts and their products
        contract_elements = root.findall('eqr:Contracts/eqr:Contract', self.namespace)
        
        for contract in contract_elements:
            contract_uid = contract.get('Uid')
            
            # Find contract products within this contract
            product_elements = contract.findall('.//eqr:ContractProduct', self.namespace)
            
            for product in product_elements:
                # Extract rate information using namespace
                rate_element = product.find('eqr:Rate', self.namespace)
                rate_description = rate_element.get('Description') if rate_element is not None else None
                rate_units = rate_element.get('Units') if rate_element is not None else None
                
                product_data = {
                    'product_uid': product.get('Uid'),
                    'contract_uid': contract_uid,
                    'product_type': product.get('ProductType'),
                    'product_name': product.get('ProductName'),
                    'product_class': product.get('Class'),
                    'term': product.get('Term'),
                    'increment': product.get('Increment'),
                    'increment_peaking': product.get('IncrementPeaking'),
                    'quantity': product.get('Quantity'),
                    'units': product.get('Units'),
                    'podsl': product.get('Podsl'),  # Point of delivery/receipt
                    'begin_date': product.get('BeginDate'),
                    'end_date': product.get('EndDate'),
                    'rate_description': rate_description,
                    'rate_units': rate_units,
                    'filing_type_product': product.get('FilingType'),  # Product-level filing type
                    'filing_uid': filing_metadata['filing_uid'],
                    'year': filing_metadata['year'],
                    'quarter': filing_metadata['quarter'],
                    'period_type': filing_metadata['period_type'],
                    'filing_type': filing_metadata['filing_type']  # Filing-level filing type
                }
                products.append(product_data)
        
        df = pd.DataFrame(products)
        if not df.empty:
            self.logger.debug(f"Parsed {len(df)} contract products")
            
            # Convert numeric columns
            numeric_columns = ['quantity']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert date columns
            date_columns = ['begin_date', 'end_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _parse_transactions(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse transactions from contracts in the XML.
        
        Transactions are nested within contracts and contain pricing, quantity,
        and timing information. This method efficiently handles large files
        with thousands of transactions.
        
        Args:
            root: Root XML element
            filing_metadata: Filing metadata dictionary
            
        Returns:
            DataFrame with transaction data
        """
        transactions = []
        
        # More efficient approach: iterate through contracts and their transactions
        contract_elements = root.findall('eqr:Contracts/eqr:Contract', self.namespace)
        
        for contract in contract_elements:
            contract_uid = contract.get('Uid')
            
            # Find transactions within this contract
            transaction_elements = contract.findall('.//eqr:Transaction', self.namespace)
            
            for transaction in transaction_elements:
                transaction_data = {
                    'transaction_uid': transaction.get('Uid'),
                    'contract_uid': contract_uid,
                    'transaction_group_ref': transaction.get('TransactionGroupRef'),
                    'begin_date': transaction.get('BeginDate'),
                    'end_date': transaction.get('EndDate'),
                    'time_zone': transaction.get('TimeZone'),
                    'trade_date': transaction.get('TradeDate'),
                    'podba': transaction.get('Podba'),
                    'podsl': transaction.get('Podsl'),
                    'transaction_class': transaction.get('Class'),
                    'term': transaction.get('Term'),
                    'increment': transaction.get('Increment'),
                    'increment_peaking': transaction.get('IncrementPeaking'),
                    'product_name': transaction.get('ProductName'),
                    'quantity': transaction.get('Quantity'),
                    'standardized_quantity': transaction.get('StandardizedQuantity'),
                    'price': transaction.get('Price'),
                    'standardized_price': transaction.get('StandardizedPrice'),
                    'rate_units': transaction.get('RateUnits'),
                    'rate_type': transaction.get('RateType'),
                    'total_transmission_charge': transaction.get('TotalTransmissionCharge'),
                    'transaction_charge': transaction.get('TransactionCharge'),
                    'filing_type_transaction': transaction.get('FilingType'),  # Transaction-level filing type
                    'filing_uid': filing_metadata['filing_uid'],
                    'year': filing_metadata['year'],
                    'quarter': filing_metadata['quarter'],
                    'period_type': filing_metadata['period_type'],
                    'filing_type': filing_metadata['filing_type']  # Filing-level filing type
                }
                transactions.append(transaction_data)
        
        df = pd.DataFrame(transactions)
        if not df.empty:
            self.logger.debug(f"Parsed {len(df)} transactions")
            
            # Convert numeric columns
            numeric_columns = [
                'quantity', 'standardized_quantity', 'price', 'standardized_price',
                'total_transmission_charge', 'transaction_charge'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert date columns
            date_columns = ['begin_date', 'end_date', 'trade_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def parse_multiple_files(self, xml_files: List[str]) -> Dict[str, List[pd.DataFrame]]:
        """Parse multiple XML files and combine results.
        
        Args:
            xml_files: List of XML file paths
            
        Returns:
            Dictionary mapping table names to lists of DataFrames
        """
        all_dataframes = {}
        successful_files = 0
        failed_files = 0
        
        self.logger.info(f"Parsing {len(xml_files)} EQR XML files")
        
        for xml_file in xml_files:
            try:
                dataframes = self.parse_xml_file(xml_file)
                
                # Merge dataframes by table name
                for table_name, df in dataframes.items():
                    if table_name not in all_dataframes:
                        all_dataframes[table_name] = []
                    all_dataframes[table_name].append(df)
                
                successful_files += 1
                
            except Exception as e:
                self.logger.error(f"Failed to parse {xml_file}: {e}")
                failed_files += 1
                continue
        
        self.logger.info(
            f"Parsing complete: {successful_files} successful, {failed_files} failed. "
            f"Found {len(all_dataframes)} table types."
        )
        
        return all_dataframes
    
    def get_parsing_stats(self) -> Dict[str, int]:
        """Get statistics about parsing performance.
        
        Returns:
            Dictionary with parsing statistics
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset parsing statistics."""
        for key in self.stats:
            self.stats[key] = 0